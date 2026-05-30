"""
utils.py — Funciones auxiliares del sistema LSM.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Contiene:
  A. Detección MediaPipe y dibujo de landmarks.
  B. Extracción y concatenación de keypoints (pose + cara + manos).
  C. Manejo de imágenes PNG con transparencia (logos).
  D. Componentes gráficos de la UI: barra de predicción, buffer,
     estabilidad y constructor de oraciones.
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

from config import (
    USE_FULL_FACE, FACE_KEY_INDICES,
    POSE_DIM, FACE_FULL_DIM, FACE_KEY_DIM,
    LEFT_HAND_DIM, RIGHT_HAND_DIM, FEATURE_DIM,
    LOGO_HEIGHT, SEQUENCE_LENGTH, STABILITY_FRAMES,
)

# ── Aliases de MediaPipe ────────────────────────────────────────────────────
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ══════════════════════════════════════════════════════════════════════════════
# A.  MEDIAPIPE — DETECCIÓN Y DIBUJO
# ══════════════════════════════════════════════════════════════════════════════

def mediapipe_detection(image: np.ndarray, model) -> tuple[np.ndarray, object]:
    """
    Procesa un frame BGR con MediaPipe Holistic con mínima copia de memoria.

    Pasos:
      1. BGR → RGB  (in-place, sin copia extra).
      2. Deshabilitar writeable para que MediaPipe evite copias internas.
      3. Procesar.
      4. Restaurar writeable y reconvertir RGB → BGR.

    Returns:
        image  : Frame BGR listo para dibujar.
        results: Objeto con face_landmarks, pose_landmarks,
                 left_hand_landmarks, right_hand_landmarks.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image: np.ndarray, results) -> None:
    """
    Dibuja todos los landmarks con estilos de color diferenciados por región.

    Colores:
      Cara              → estilo predeterminado de MediaPipe (malla completa)
      Pose              → rojo oscuro / morado
      Mano izquierda    → rosa / azul violeta
      Mano derecha      → naranja / fucsia
    """
    # Cara
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

    # Pose (esqueleto completo)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10),  thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

    # Mano izquierda
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

    # Mano derecha
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


# ══════════════════════════════════════════════════════════════════════════════
# B.  EXTRACCIÓN DE KEYPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _landmarks_to_array(landmark_list, zero_size: int,
                        indices: list[int] | None = None,
                        n_coords: int = 3) -> np.ndarray:
    """
    Convierte una NormalizedLandmarkList a un vector NumPy float32.

    Si landmark_list es None (región no detectada), devuelve un vector
    de ceros del tamaño correcto para mantener la dimensión fija.

    Args:
        landmark_list: Resultado de MediaPipe para una región (o None).
        zero_size:     Tamaño del vector de ceros cuando no hay detección.
        indices:       Lista de índices a extraer. None = todos los puntos.
        n_coords:      3 → (x,y,z)  |  4 → (x,y,z,visibility).
    Returns:
        np.ndarray de shape (zero_size,) o (len(indices)*n_coords,), dtype float32.
    """
    if landmark_list is None:
        return np.zeros(zero_size, dtype=np.float32)

    lm = landmark_list.landmark
    pts = [lm[i] for i in indices] if indices is not None else list(lm)

    if n_coords == 4:
        arr = np.array([[p.x, p.y, p.z, p.visibility] for p in pts], dtype=np.float32)
    else:
        arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)

    return arr.flatten()


def extract_keypoints(results) -> np.ndarray:
    """
    Extrae y concatena todos los landmarks en un único vector 1-D por frame.

    Estructura del vector de salida
    ────────────────────────────────
    Modo completo   (USE_FULL_FACE=True)  → 1 662 valores
      [  0: 131] Pose         33 pts × 4 coords  (x,y,z,vis)
      [132:1535] Cara         468 pts × 3 coords  (x,y,z)
      [1536:1598] Mano izq.  21 pts × 3 coords
      [1599:1661] Mano der.  21 pts × 3 coords

    Modo reducido  (USE_FULL_FACE=False) → 492 valores
      [  0: 131] Pose         33 pts × 4 coords
      [132: 365] Cara key     78 pts × 3 coords  (labios+cejas+ojos+nariz)
      [366: 428] Mano izq.   21 pts × 3 coords
      [429: 491] Mano der.   21 pts × 3 coords

    Si una región no está detectada, su segmento se rellena con ceros.

    Returns:
        keypoints: np.ndarray shape (FEATURE_DIM,), dtype float32.
    """
    pose = _landmarks_to_array(
        results.pose_landmarks, zero_size=POSE_DIM,
        indices=None, n_coords=4,
    )

    if USE_FULL_FACE:
        face = _landmarks_to_array(
            results.face_landmarks, zero_size=FACE_FULL_DIM,
            indices=None, n_coords=3,
        )
    else:
        face = _landmarks_to_array(
            results.face_landmarks, zero_size=FACE_KEY_DIM,
            indices=FACE_KEY_INDICES, n_coords=3,
        )

    left_hand = _landmarks_to_array(
        results.left_hand_landmarks, zero_size=LEFT_HAND_DIM,
        indices=None, n_coords=3,
    )
    right_hand = _landmarks_to_array(
        results.right_hand_landmarks, zero_size=RIGHT_HAND_DIM,
        indices=None, n_coords=3,
    )

    keypoints = np.concatenate([pose, face, left_hand, right_hand])

    assert keypoints.shape[0] == FEATURE_DIM, (
        f"[ERROR] Dimensión incorrecta: obtenida={keypoints.shape[0]}, "
        f"esperada={FEATURE_DIM}. "
        f"Verifica USE_FULL_FACE en config.py y vuelve a recolectar datos."
    )
    return keypoints


def hands_detected(results) -> bool:
    """Devuelve True si al menos una mano es visible en el frame actual."""
    return (
        results.left_hand_landmarks  is not None or
        results.right_hand_landmarks is not None
    )


# ══════════════════════════════════════════════════════════════════════════════
# C.  MANEJO DE IMÁGENES PNG CON TRANSPARENCIA
# ══════════════════════════════════════════════════════════════════════════════

def load_logo(path: str, target_height: int) -> np.ndarray | None:
    """
    Carga un logo PNG (con o sin canal alfa) y lo redimensiona
    manteniendo la relación de aspecto original.

    Args:
        path:          Ruta al archivo PNG.
        target_height: Altura objetivo en píxeles.
    Returns:
        Imagen BGRA redimensionada, o None si el archivo no existe.
    """
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        print(f"[ADVERTENCIA] Logo no encontrado: '{path}'  "
              f"→ coloca tu PNG en la carpeta assets/")
        return None

    # Agregar canal alfa opaco si la imagen solo tiene 3 canales
    if logo.ndim == 3 and logo.shape[2] == 3:
        alpha = np.full((*logo.shape[:2], 1), 255, dtype=np.uint8)
        logo  = np.concatenate([logo, alpha], axis=2)

    h, w  = logo.shape[:2]
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(logo, (new_w, target_height), interpolation=cv2.INTER_AREA)


def overlay_png(background: np.ndarray, overlay: np.ndarray,
                x: int, y: int) -> np.ndarray:
    """
    Superpone una imagen BGRA sobre un fondo BGR usando el canal alfa.
    Opera in-place sobre background; maneja automáticamente recorte en bordes.

    Args:
        background: Imagen de fondo  (H×W×3, BGR, uint8).
        overlay:    Imagen a poner   (h×w×4, BGRA, uint8).
        x, y:       Esquina superior izquierda donde colocar el overlay.
    Returns:
        background (mismo objeto, modificado in-place).
    """
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    # Calcular región válida de intersección
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ov_w, bg_w), min(y + ov_h, bg_h)
    if x2 <= x1 or y2 <= y1:
        return background

    # Regiones recortadas
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    roi    = background[y1:y2, x1:x2].astype(np.float32)
    ov_roi = overlay[oy1:oy2, ox1:ox2]

    # Composición alfa (A over B)
    alpha   = ov_roi[:, :, 3:4].astype(np.float32) / 255.0
    bgr     = ov_roi[:, :, :3].astype(np.float32)
    blended = bgr * alpha + roi * (1.0 - alpha)

    background[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)
    return background


# ══════════════════════════════════════════════════════════════════════════════
# D.  COMPONENTES GRÁFICOS DE LA UI
# ══════════════════════════════════════════════════════════════════════════════

def draw_prediction_box(image: np.ndarray, text: str, confidence: float,
                        color: tuple[int, int, int] = (0, 230, 0)) -> None:
    """
    Panel inferior semitransparente con:
      · Nombre de la seña confirmada (texto grande y centrado).
      · Barra de confianza horizontal con porcentaje.
      · Línea de acento de color en el borde superior del panel.
    """
    h, w    = image.shape[:2]
    panel_h = 108

    # Fondo semitransparente
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)

    # Línea de acento de color
    cv2.line(image, (0, h - panel_h), (w, h - panel_h), color, 3)

    # Texto centrado (seña)
    label       = text.upper().replace('_', ' ')
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 1.9
    thickness   = 3
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tx          = max((w - tw) // 2, 10)
    cv2.putText(image, label, (tx, h - 55), font, font_scale, color, thickness, cv2.LINE_AA)

    # Barra de confianza
    bx1, by1 = 20, h - 22
    bx2       = w - 110
    bar_w     = int((bx2 - bx1) * confidence)
    cv2.rectangle(image, (bx1, by1), (bx2, by1 + 10), (45, 45, 45), -1)
    cv2.rectangle(image, (bx1, by1), (bx1 + bar_w, by1 + 10), color, -1)

    # Porcentaje numérico
    pct = f"{confidence:.1%}"
    cv2.putText(image, pct, (bx2 + 6, by1 + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)


def draw_buffer_indicator(image: np.ndarray,
                          current_len: int, max_len: int) -> None:
    """
    Barra horizontal delgada que muestra el llenado del buffer de secuencia.
      · Naranja → cargando frames.
      · Verde   → buffer completo, modelo puede predecir.
    """
    h, w   = image.shape[:2]
    bar_y  = h - 120
    bar_h  = 7
    filled = int(w * current_len / max_len)
    color  = (0, 230, 0) if current_len >= max_len else (0, 170, 255)

    cv2.rectangle(image, (0, bar_y), (w, bar_y + bar_h), (30, 30, 30), -1)
    cv2.rectangle(image, (0, bar_y), (filled, bar_y + bar_h), color, -1)


def draw_stability_bar(image: np.ndarray,
                       stable_count: int, stability_frames: int) -> None:
    """
    Indicador de estabilidad: muestra cuántos frames consecutivos lleva
    la misma predicción. Cambia de rojo a verde al acercarse al umbral.
    """
    h, w = image.shape[:2]
    ratio = min(stable_count / stability_frames, 1.0)

    # Gradiente rojo → verde
    r = int(240 * (1.0 - ratio))
    g = int(240 * ratio)
    color = (0, g, r)

    label = f"Estabilidad: {stable_count}/{stability_frames}"
    cv2.putText(image, label, (15, h - 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def draw_sentence_builder(image: np.ndarray, words: list[str]) -> None:
    """
    Muestra las palabras confirmadas en la parte superior central del frame,
    formando una "oración" acumulada visible durante toda la sesión.
    Muestra como máximo las últimas 6 palabras para no saturar la UI.
    """
    if not words:
        return

    h, w    = image.shape[:2]
    sentence = '   '.join(w.upper().replace('_', ' ') for w in words[-6:])
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.80
    thick    = 2
    text_y   = LOGO_HEIGHT + 52

    (tw, th), _ = cv2.getTextSize(sentence, font, scale, thick)
    tx   = max((w - tw) // 2, 10)
    pad  = 10

    # Fondo semitransparente
    overlay = image.copy()
    cv2.rectangle(overlay,
                  (tx - pad, text_y - th - pad),
                  (tx + tw + pad, text_y + pad),
                  (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    # Borde
    cv2.rectangle(image,
                  (tx - pad, text_y - th - pad),
                  (tx + tw + pad, text_y + pad),
                  (80, 80, 80), 1)

    # Texto
    cv2.putText(image, sentence, (tx, text_y), font, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
