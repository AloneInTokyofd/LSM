"""
main.py — Sistema de Reconocimiento de LSM Dinámica en Tiempo Real.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Laboratorio de Inteligencia Artificial (LIA) - CECyT 16, IPN.

DESCRIPCIÓN
───────────
Captura video en tiempo real, extrae landmarks con MediaPipe Holistic,
acumula secuencias de 30 frames en un buffer circular, ejecuta el modelo
LSTM y aplica lógica de umbral de confianza + estabilidad temporal antes
de confirmar y mostrar una predicción.

CONTROLES
──────────
  Q / ESC → Salir limpiamente.
  R       → Resetear buffer y estado de predicción.y l
  C       → Limpiar la oración construida en pantalla.
  S       → Guardar la oración actual en sesiones/<timestamp>.txt.

EJECUCIÓN
──────────
  python main.py
  (Requiere haber entrenado el modelo con train.py)
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import time
import os
from collections import deque
from datetime import datetime

from config import (
    SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, STABILITY_FRAMES,
    COOLDOWN_SECONDS, MODEL_PATH, LABEL_MAP_PATH, ACTIONS,
    LOGO_LIA_PATH, LOGO_CECYT_PATH, LOGO_HEIGHT,
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, MODELS_DIR, SESSIONS_DIR,
    FEATURE_DIM,
)
from utils import (
    mediapipe_detection, draw_styled_landmarks, extract_keypoints,
    overlay_png, load_logo,
    draw_prediction_box, draw_buffer_indicator,
    draw_stability_bar, draw_sentence_builder,
    hands_detected,
)


# ══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class LSMRecognizer:
    """
    Sistema completo de reconocimiento de LSM dinámica en tiempo real.

    Responsabilidades
    ─────────────────
    · Inicializa y carga el modelo LSTM desde disco.
    · Mantiene un buffer circular (deque) de SEQUENCE_LENGTH frames.
    · Extrae keypoints con MediaPipe Holistic por cada frame.
    · Ejecuta inferencia cuando el buffer está completo.
    · Aplica lógica de doble umbral:
        a) Confianza  ≥ CONFIDENCE_THRESHOLD
        b) Misma predicción durante ≥ STABILITY_FRAMES consecutivos
    · Acumula palabras confirmadas en una "oración" en pantalla.
    · Renderiza UI completa: logos, buffer, estabilidad, predicción, FPS.
    """

    # ── Constantes de UI ───────────────────────────────────────────
    _COLOR_CONFIRMED   = (0,  230,   0)    # Verde  — predicción confirmada
    _COLOR_TENTATIVE   = (130, 130, 130)   # Gris   — predicción tentativa
    _COLOR_BUFFER_FULL = (0,  200, 100)    # Verde suave — buffer lleno
    _COLOR_ACCENT      = (0,  170, 255)    # Azul   — elementos de UI

    def __init__(self) -> None:
        # ── Modelo y etiquetas ─────────────────────────────────────
        self.model:     tf.keras.Model | None = None
        self.label_map: dict[int, str]        = {}
        self._load_model()

        # ── Buffer circular de secuencia temporal ──────────────────
        # maxlen garantiza que al añadir el frame N+1 se descarta el 0,
        # manteniendo siempre los SEQUENCE_LENGTH frames más recientes.
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

        # ── Estado de predicción en curso ──────────────────────────
        self.current_label:      str   = ""
        self.current_confidence: float = 0.0

        # ── Estado de estabilidad (doble umbral) ───────────────────
        self.stable_label: str = ""
        self.stable_count: int = 0

        # ── Predicción confirmada (visible en pantalla) ────────────
        self.confirmed_word:       str   = ""
        self.confirmed_confidence: float = 0.0
        self._last_confirmed:      str   = ""      # Anti-duplicado
        self._confirm_cooldown:    float = 0.0     # Timestamp de siguiente confirmación

        # ── Constructor de oraciones ───────────────────────────────
        self.sentence_words: list[str] = []

        # ── Activos de UI ──────────────────────────────────────────
        self.logo_lia = load_logo(LOGO_LIA_PATH, LOGO_HEIGHT)
        self.logo_cec = load_logo(LOGO_CECYT_PATH, LOGO_HEIGHT)

        # ── FPS suavizado (ventana de 30 timestamps) ───────────────
        self._fps_times: deque[float] = deque(maxlen=30)

        # ── Inferencia: ejecutar solo cada N frames ────────────────
        # Valor 1 = cada frame. Aumentar (2 o 3) si la CPU no da abasto.
        self._predict_every: int  = 1
        self._frame_count:   int  = 0

    # ══════════════════════════════════════════════════════════════
    # CARGA DEL MODELO
    # ══════════════════════════════════════════════════════════════

    def _load_model(self) -> None:
        """
        Carga el modelo LSTM y el mapa de etiquetas desde disco.
        Si el modelo no existe, el sistema funciona en modo demo
        (sin predicciones) para probar la UI y la detección.
        """
        if not os.path.exists(MODEL_PATH):
            print(f'[ADVERTENCIA] Modelo no encontrado: {MODEL_PATH}')
            print('              Ejecuta train.py para generar el modelo.')
            print('              El sistema arrancará en modo DEMO (sin predicciones).')
            self.label_map = {i: a for i, a in enumerate(ACTIONS)}
            return

        print(f'[INFO] Cargando modelo: {MODEL_PATH}')
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # Re-compilar sin optimizer checkpoint para inferencia pura
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        # Warm-up: elimina la latencia de la primera predicción real
        dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print('[INFO] Modelo calentado y listo para inferencia.')

        # Cargar label map
        if os.path.exists(LABEL_MAP_PATH):
            with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self.label_map = {int(k): v for k, v in raw.items()}
        else:
            print(f'[WARN] label_map.json no encontrado. Usando ACTIONS de config.py.')
            self.label_map = {i: a for i, a in enumerate(ACTIONS)}

        print(f'[INFO] Clases cargadas ({len(self.label_map)}): '
              f'{list(self.label_map.values())}')

    # ══════════════════════════════════════════════════════════════
    # LÓGICA DE INFERENCIA Y ESTABILIZACIÓN
    # ══════════════════════════════════════════════════════════════

    def _run_inference(self) -> None:
        """
        Ejecuta una pasada del modelo LSTM sobre el buffer actual.

        Lógica de doble umbral
        ──────────────────────
        Paso 1 — Umbral de confianza:
          La predicción solo es candidata si su probabilidad supera
          CONFIDENCE_THRESHOLD. Bajo el umbral → resetear estabilidad.

        Paso 2 — Estabilidad temporal:
          El mismo label debe mantenerse durante STABILITY_FRAMES
          fotogramas consecutivos para ser confirmado.

        Paso 3 — Cooldown anti-duplicado:
          Después de confirmar una seña, se impone un silencio de
          COOLDOWN_SECONDS y no se acepta la misma seña de nuevo hasta
          que cambie o pase el cooldown.

        Este mecanismo elimina los "parpadeos" (flickering) que ocurren
        cuando el modelo oscila entre dos clases similares.
        """
        if len(self.frame_buffer) < SEQUENCE_LENGTH or self.model is None:
            return

        # Construir tensor (1, SEQUENCE_LENGTH, FEATURE_DIM)
        seq   = np.stack(self.frame_buffer, axis=0)[np.newaxis, ...].astype(np.float32)
        probs = self.model.predict(seq, verbose=0)[0]   # shape: (num_classes,)

        best_idx  = int(np.argmax(probs))
        best_conf = float(probs[best_idx])
        best_lbl  = self.label_map.get(best_idx, f'clase_{best_idx}')

        # Actualizar estado de predicción en curso
        self.current_label      = best_lbl
        self.current_confidence = best_conf

        # ── Paso 1: Umbral de confianza ───────────────────────────
        if best_conf < CONFIDENCE_THRESHOLD:
            self.stable_label = ""
            self.stable_count = 0
            return

        # ── Paso 2: Estabilidad temporal ──────────────────────────
        if best_lbl == self.stable_label:
            self.stable_count += 1
        else:
            self.stable_label = best_lbl
            self.stable_count = 1

        # ── Paso 3: Confirmación con cooldown ────────────────────
        if self.stable_count >= STABILITY_FRAMES:
            now = time.monotonic()
            if (now >= self._confirm_cooldown and
                    best_lbl != self._last_confirmed):
                self.confirmed_word       = best_lbl
                self.confirmed_confidence = best_conf
                self._last_confirmed      = best_lbl
                self._confirm_cooldown    = now + COOLDOWN_SECONDS

                self.sentence_words.append(best_lbl)
                print(f'  [✓] Confirmada: "{best_lbl}"  '
                      f'({best_conf:.1%})  '
                      f'frames estables: {self.stable_count}')

    def _reset_state(self) -> None:
        """Vacía el buffer y resetea todo el estado de predicción."""
        self.frame_buffer.clear()
        self.current_label      = ""
        self.current_confidence = 0.0
        self.stable_label       = ""
        self.stable_count       = 0
        self.confirmed_word     = ""
        self.confirmed_confidence = 0.0
        self._last_confirmed    = ""
        print('[INFO] Buffer y estado de predicción reseteados.')

    # ══════════════════════════════════════════════════════════════
    # FPS
    # ══════════════════════════════════════════════════════════════

    def _update_fps(self) -> float:
        """
        Calcula FPS suavizado usando una ventana deslizante de 30 frames.
        Más estable que 1 / delta_t entre dos frames consecutivos.
        """
        self._fps_times.append(time.perf_counter())
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0

    # ══════════════════════════════════════════════════════════════
    # RENDERIZADO DE UI
    # ══════════════════════════════════════════════════════════════

    def _render_ui(self, image: np.ndarray, fps: float) -> None:
        """
        Dibuja todos los elementos gráficos sobre el frame actual.

        Elementos (de arriba a abajo):
          · Logo LIA           — esquina superior izquierda.
          · Logo CECyT 16      — esquina superior derecha.
          · Oración construida — franja central superior.
          · Instrucciones      — debajo de los logos.
          · Estado del buffer  — esquina superior derecha (texto).
          · FPS                — esquina inferior derecha (texto).
          · Barra de estabilidad — zona inferior media.
          · Barra de buffer    — línea horizontal cerca del panel.
          · Panel de predicción — franja inferior (si hay confirmación).
          · Predicción tentativa — texto gris si hay candidato no confirmado.
        """
        h, w = image.shape[:2]

        # ── Logos ─────────────────────────────────────────────────
        if self.logo_lia is not None:
            overlay_png(image, self.logo_lia, 10, 10)
        if self.logo_cec is not None:
            x_right = w - self.logo_cec.shape[1] - 10
            overlay_png(image, self.logo_cec, x_right, 10)

        # ── Oración acumulada (franja superior central) ───────────
        draw_sentence_builder(image, self.sentence_words)

        # ── Atajos de teclado ─────────────────────────────────────
        hints = 'Q=Salir   R=Reset   C=Limpiar   S=Guardar'
        cv2.putText(image, hints, (15, LOGO_HEIGHT + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (110, 110, 110), 1, cv2.LINE_AA)

        # ── Estado del buffer (texto, esquina superior derecha) ────
        buf_full  = len(self.frame_buffer) >= SEQUENCE_LENGTH
        buf_color = self._COLOR_BUFFER_FULL if buf_full else self._COLOR_ACCENT
        buf_txt   = f'Buffer: {len(self.frame_buffer)}/{SEQUENCE_LENGTH}'
        cv2.putText(image, buf_txt, (w - 210, LOGO_HEIGHT + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, buf_color, 1, cv2.LINE_AA)

        # ── FPS (esquina inferior derecha) ────────────────────────
        cv2.putText(image, f'FPS: {fps:.1f}', (w - 130, h - 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (170, 170, 170), 1, cv2.LINE_AA)

        # ── Barra de estabilidad ──────────────────────────────────
        if self.stable_count > 0:
            draw_stability_bar(image, self.stable_count, STABILITY_FRAMES)

        # ── Barra de llenado del buffer ───────────────────────────
        draw_buffer_indicator(image, len(self.frame_buffer), SEQUENCE_LENGTH)

        # ── Predicción confirmada o tentativa ────────────────────
        if self.confirmed_word:
            draw_prediction_box(
                image, self.confirmed_word,
                self.confirmed_confidence,
                color=self._COLOR_CONFIRMED,
            )
        elif (self.current_label and
              self.current_confidence >= CONFIDENCE_THRESHOLD * 0.72):
            # Predicción candidata aún no estable → mostrar en gris
            txt = (f'? {self.current_label.upper().replace("_", " ")}'
                   f'  ({self.current_confidence:.0%})')
            cv2.putText(image, txt, (20, h - 132),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        self._COLOR_TENTATIVE, 1, cv2.LINE_AA)

        # ── Indicador "SIN MODELO" ────────────────────────────────
        if self.model is None:
            cv2.putText(image, 'MODO DEMO — Sin modelo cargado',
                        (20, h - 132), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 100, 255), 2, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════
    # GUARDADO DE SESIÓN
    # ══════════════════════════════════════════════════════════════

    def _save_session(self) -> None:
        """
        Guarda la oración acumulada en un archivo .txt con timestamp.
        Ruta: sesiones/sesion_YYYYMMDD_HHMMSS.txt
        """
        if not self.sentence_words:
            print('[INFO] No hay palabras para guardar.')
            return

        os.makedirs(SESSIONS_DIR, exist_ok=True)
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(SESSIONS_DIR, f'sesion_{ts}.txt')
        sentence = ' '.join(self.sentence_words)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f'Fecha    : {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
            f.write(f'Palabras : {len(self.sentence_words)}\n')
            f.write(f'Oración  : {sentence}\n')

        print(f'[INFO] Sesión guardada → {filepath}')
        print(f'       Contenido : "{sentence}"')

    # ══════════════════════════════════════════════════════════════
    # LOOP PRINCIPAL
    # ══════════════════════════════════════════════════════════════

    def run(self) -> None:
        """
        Bucle principal de captura, procesamiento y visualización.

        Flujo por frame:
          1. Leer frame de la cámara.
          2. Detectar landmarks con MediaPipe Holistic.
          3. Extraer keypoints y añadir al buffer circular.
          4. Si hay manos detectadas, ejecutar inferencia (cada _predict_every frames).
          5. Dibujar landmarks sobre el frame.
          6. Renderizar UI completa.
          7. Mostrar ventana y procesar teclas.
        """
        # ── Configurar cámara ─────────────────────────────────────
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # Minimizar latencia del buffer HW

        if not cap.isOpened():
            raise RuntimeError(
                f'No se pudo abrir la cámara (índice {CAMERA_INDEX}). '
                'Verifica CAMERA_INDEX en config.py.'
            )

        # ── Configurar ventana ────────────────────────────────────
        window = 'LSM — Reconocimiento en Tiempo Real | LIA · CECyT 16'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)

        print('\n[INFO] Sistema LSM iniciado.')
        print('[INFO] Controles: Q/ESC=Salir  R=Reset  C=Limpiar  S=Guardar')
        print('─' * 60)

        # ── MediaPipe Holistic ────────────────────────────────────
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,        # 0=rápido · 1=equilibrado · 2=preciso
            smooth_landmarks=True,
        ) as holistic:

            while True:
                ret, frame = cap.read()
                if not ret:
                    print('[ERROR] No se puede leer el frame. '
                          '¿Cámara desconectada?')
                    break

                # Mostrar como espejo horizontal: la mano derecha del usuario
                # queda en el lado derecho de la ventana. Aplicamos el mismo
                # cambio antes de MediaPipe para mantener consistencia espacial.
                frame = cv2.flip(frame, 1)

                # ── Detección MediaPipe ───────────────────────────
                image, results = mediapipe_detection(frame, holistic)

                # ── Keypoints → buffer ────────────────────────────
                keypoints = extract_keypoints(results)
                self.frame_buffer.append(keypoints)

                # ── Inferencia (opcional: solo si hay manos) ──────
                self._frame_count += 1
                if (hands_detected(results) and
                        self._frame_count % self._predict_every == 0):
                    self._run_inference()

                # ── Dibujo de landmarks ───────────────────────────
                draw_styled_landmarks(image, results)

                # ── Cálculo de FPS y UI ───────────────────────────
                fps = self._update_fps()
                self._render_ui(image, fps)

                cv2.imshow(window, image)

                # ── Manejo de teclas ──────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):          # Q / ESC → salir
                    break
                elif key == ord('r'):              # R → resetear
                    self._reset_state()
                elif key == ord('c'):              # C → limpiar oración
                    self.sentence_words.clear()
                    self.confirmed_word = ""
                    print('[INFO] Oración limpiada.')
                elif key == ord('s'):              # S → guardar sesión
                    self._save_session()

        cap.release()
        cv2.destroyAllWindows()
        print('\n[INFO] Sistema cerrado correctamente.')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Punto de entrada: configura GPU y lanza el reconocedor."""
    print('=' * 60)
    print('  Sistema LSM — Reconocimiento Dinámico en Tiempo Real')
    print('  Laboratorio de Inteligencia Artificial (LIA)')
    print('  CECyT 16, Instituto Politécnico Nacional')
    print('=' * 60)

    # Configurar crecimiento dinámico de memoria GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'[INFO] GPU detectada: {[g.name for g in gpus]}')
    else:
        print('[INFO] GPU no detectada — usando CPU.')

    LSMRecognizer().run()


if __name__ == '__main__':
    main()
