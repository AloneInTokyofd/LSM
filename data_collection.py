"""
data_collection.py — Recolección de secuencias de landmarks para LSM dinámica.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Laboratorio de Inteligencia Artificial (LIA) - CECyT 16, IPN.

FLUJO GENERAL
─────────────
  1. Crea automáticamente la estructura de carpetas en LSM_Dataset/.
  2. Para cada seña, muestra una pantalla de espera hasta que el usuario
     presiona SPACE para indicar que está listo.
  3. Captura SEQUENCE_LENGTH frames, extrae keypoints con MediaPipe Holistic
     y los guarda como archivos .npy individuales.
  4. Repite para cada secuencia de cada seña.

CONTROLES DURANTE LA RECOLECCIÓN
──────────────────────────────────
  SPACE → Iniciar la siguiente secuencia (en pantalla de espera).
  R     → Repetir la secuencia actual si algo salió mal.
  Q/ESC → Abortar y guardar lo recolectado hasta ese punto.

ESTRUCTURA DE CARPETAS GENERADA
─────────────────────────────────
  LSM_Dataset/
  ├── hola/
  │   ├── 0/
  │   │   ├── frame_000.npy   ← Keypoints del frame 0  (shape: FEATURE_DIM,)
  │   │   ├── frame_001.npy
  │   │   └── ... (SEQUENCE_LENGTH archivos)
  │   ├── 1/
  │   └── ... (NUM_SEQUENCES carpetas)
  ├── gracias/
  └── ...

EJECUCIÓN
──────────
  python data_collection.py
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import os
import time

from config import (
    ACTIONS, SEQUENCE_LENGTH, NUM_SEQUENCES, START_FOLDER,
    DATA_PATH, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    LOGO_HEIGHT, LOGO_LIA_PATH, LOGO_CECYT_PATH,
    MODELS_DIR, ASSETS_DIR, SESSIONS_DIR, FEATURE_DIM,
)
from utils import (
    mediapipe_detection, draw_styled_landmarks, extract_keypoints,
    load_logo, overlay_png,
)


# ══════════════════════════════════════════════════════════════════════════════
# A.  CREACIÓN DE ESTRUCTURA DE CARPETAS
# ══════════════════════════════════════════════════════════════════════════════

def create_folder_structure() -> None:
    """
    Crea todos los directorios necesarios para el dataset y el proyecto.

    Árbol generado:
      LSM_Dataset/<seña>/<secuencia>/   por cada seña y número de secuencia.
      models/                           para el modelo entrenado.
      assets/                           para logos PNG.
      sesiones/                         para oraciones guardadas.
    """
    for folder in [MODELS_DIR, ASSETS_DIR, SESSIONS_DIR]:
        os.makedirs(folder, exist_ok=True)

    total = 0
    for action in ACTIONS:
        for seq_idx in range(START_FOLDER, START_FOLDER + NUM_SEQUENCES):
            path = os.path.join(DATA_PATH, action, str(seq_idx))
            os.makedirs(path, exist_ok=True)
            total += 1

    print('┌' + '─' * 52 + '┐')
    print('│       ESTRUCTURA DE CARPETAS CREADA             │')
    print('├' + '─' * 52 + '┤')
    print(f'│  Señas        : {len(ACTIONS):<34} │')
    print(f'│  Secuencias   : {NUM_SEQUENCES:<34} │')
    print(f'│  Carpetas tot.: {total:<34} │')
    print(f'│  FEATURE_DIM  : {FEATURE_DIM:<34} │')
    print(f'│  Ruta base    : {DATA_PATH[:34]:<34} │')
    print('└' + '─' * 52 + '┘')


# ══════════════════════════════════════════════════════════════════════════════
# B.  FUNCIONES DE DIBUJO DE UI PARA RECOLECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _draw_waiting_screen(image: np.ndarray, action: str,
                         seq_idx: int, total_seqs: int,
                         logo_lia, logo_cec,
                         global_progress: float) -> None:
    """
    Pantalla de espera entre secuencias. El usuario debe presionar SPACE.
    Muestra: seña actual, número de secuencia, barra de progreso global.
    """
    h, w = image.shape[:2]

    # Oscurecer ligeramente el fondo para resaltar la UI
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

    # Logos
    if logo_lia is not None:
        overlay_png(image, logo_lia, 10, 10)
    if logo_cec is not None:
        overlay_png(image, logo_cec, w - logo_cec.shape[1] - 10, 10)

    # Cuadro central de información
    texts = [
        (f"SEÑA: {action.upper().replace('_', ' ')}", 1.5, (0, 240, 120), 3),
        (f"Secuencia {seq_idx + 1} de {total_seqs}",  0.90, (210, 210, 210), 2),
        ('',                                           0.50, (0, 0, 0),       1),
        ('Prepárate y presiona  SPACE  para iniciar',  0.70, (0, 200, 255),   2),
        ('R = Repetir anterior  |  Q = Salir',         0.55, (150, 150, 150), 1),
    ]

    y0 = h // 2 - 90
    for text, scale, color, thick in texts:
        if not text:
            y0 += 15
            continue
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        tx = max((w - tw) // 2, 10)
        cv2.putText(image, text, (tx, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y0 += int(scale * 52) + 12

    # Barra de progreso global (parte inferior)
    bx1, by = 20, h - 25
    bx2     = w - 20
    filled  = int((bx2 - bx1) * global_progress)
    cv2.rectangle(image, (bx1, by), (bx2, by + 10), (40, 40, 40), -1)
    cv2.rectangle(image, (bx1, by), (bx1 + filled, by + 10), (0, 200, 255), -1)
    pct_txt = f"Progreso global: {global_progress:.1%}"
    cv2.putText(image, pct_txt, (bx1, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def _draw_recording_ui(image: np.ndarray, action: str,
                       seq_idx: int, frame_num: int, total_seqs: int,
                       logo_lia, logo_cec) -> None:
    """
    UI durante la grabación activa de una secuencia.
    Muestra: seña, número de secuencia, frame actual, barra de progreso del frame.
    """
    h, w = image.shape[:2]

    # Logos
    if logo_lia is not None:
        overlay_png(image, logo_lia, 10, 10)
    if logo_cec is not None:
        overlay_png(image, logo_cec, w - logo_cec.shape[1] - 10, 10)

    # Panel inferior
    panel_overlay = image.copy()
    cv2.rectangle(panel_overlay, (0, h - 75), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(panel_overlay, 0.70, image, 0.30, 0, image)

    # Línea de acento roja (indica grabación activa)
    cv2.line(image, (0, h - 75), (w, h - 75), (0, 0, 220), 3)

    # ● REC
    cv2.circle(image, (22, h - 52), 8, (0, 0, 220), -1)
    cv2.putText(image, "GRABANDO", (38, h - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2, cv2.LINE_AA)

    # Seña
    cv2.putText(image,
                f"Seña: {action.upper().replace('_', ' ')}",
                (w // 2 - 100, h - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 0), 2, cv2.LINE_AA)

    # Frame / Secuencia
    info = f"Sec: {seq_idx + 1}/{total_seqs}   Frame: {frame_num + 1}/{SEQUENCE_LENGTH}"
    cv2.putText(image, info, (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # Barra de progreso del frame
    bx1, by  = 0, h - 80
    filled   = int(w * (frame_num + 1) / SEQUENCE_LENGTH)
    cv2.rectangle(image, (0, by), (w, by + 7), (35, 35, 35), -1)
    cv2.rectangle(image, (0, by), (filled, by + 7), (0, 200, 255), -1)


# ══════════════════════════════════════════════════════════════════════════════
# C.  LOOP PRINCIPAL DE RECOLECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def collect_data() -> None:
    """
    Pipeline completo de recolección de datos.

    Para cada seña × secuencia:
      1. Pantalla de espera hasta SPACE.
      2. Captura SEQUENCE_LENGTH frames + extrae keypoints.
      3. Guarda cada frame como .npy en la carpeta correspondiente.
      4. 'R' repite la secuencia; 'Q'/'ESC' interrumpe todo.
    """
    create_folder_structure()

    logo_lia = load_logo(LOGO_LIA_PATH, LOGO_HEIGHT)
    logo_cec = load_logo(LOGO_CECYT_PATH, LOGO_HEIGHT)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara (índice {CAMERA_INDEX}).")
        return

    window = 'LSM - Recolección de Datos | LIA - CECyT 16'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)

    total_seqs   = START_FOLDER + NUM_SEQUENCES
    total_items  = len(ACTIONS) * NUM_SEQUENCES
    done_items   = 0
    abort        = False

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        smooth_landmarks=True,
    ) as holistic:

        for action_idx, action in enumerate(ACTIONS):
            if abort:
                break

            print(f'\n{"═" * 55}')
            print(f'  [{action_idx + 1}/{len(ACTIONS)}]  Recolectando: {action.upper()}')
            print(f'{"═" * 55}')

            seq = START_FOLDER
            while seq < total_seqs:
                if abort:
                    break

                global_progress = done_items / total_items

                # ── PANTALLA DE ESPERA ──────────────────────────────
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        abort = True
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    _draw_waiting_screen(image, action, seq, total_seqs,
                                         logo_lia, logo_cec, global_progress)
                    cv2.imshow(window, image)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):         # SPACE → iniciar grabación
                        break
                    elif key in (ord('q'), 27): # Q / ESC → abortar
                        abort = True
                        break

                if abort:
                    break

                # ── GRABACIÓN DE FRAMES ─────────────────────────────
                repeat = False
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        abort = True
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    _draw_recording_ui(image, action, seq, frame_num,
                                       total_seqs, logo_lia, logo_cec)
                    cv2.imshow(window, image)

                    # Guardar keypoints del frame actual
                    keypoints = extract_keypoints(results)
                    npy_path  = os.path.join(
                        DATA_PATH, action, str(seq),
                        f'frame_{frame_num:03d}.npy'
                    )
                    np.save(npy_path, keypoints)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        abort = True
                        break
                    elif key == ord('r'):
                        print(f'  [INFO] Repitiendo secuencia {seq + 1}...')
                        repeat = True
                        break

                if abort:
                    break

                if not repeat:
                    done_items += 1
                    print(f'  ✓  {action:<20}  Sec {seq + 1}/{total_seqs}  '
                          f'({done_items}/{total_items} totales)')
                    seq += 1

    cap.release()
    cv2.destroyAllWindows()

    print()
    if abort:
        print('[INFO] Recolección interrumpida. Los datos guardados son válidos.')
    else:
        print('[✓] Recolección completada exitosamente.')
    print(f'    Dataset guardado en: {DATA_PATH}/')
    print(f'    Total secuencias   : {done_items}/{total_items}')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    collect_data()
