"""
config.py — Configuración global del sistema LSM dinámico.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Laboratorio de Inteligencia Artificial (LIA) - CECyT 16, IPN.

INSTRUCCIONES:
  · Modifica ACTIONS con las señas que vayas a reconocer.
  · Ajusta USE_FULL_FACE según tu hardware (False = CPU amigable).
  · Todos los scripts importan desde aquí — edita solo este archivo.
"""
import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SEÑAS A RECONOCER
#     Añade, elimina o renombra según tu dataset.
#     Usa guión bajo para señas de varias palabras: 'me_llamo', 'buenos_dias'.
# ═══════════════════════════════════════════════════════════════════════════════
ACTIONS = np.array([
    'hola',
    'gracias',
    'por_favor',
    'me_llamo',
    'mucho_gusto',
    'buenos_dias',
    'adios',
    'si',
    'no',
    'ayuda',
])

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PARÁMETROS DE SECUENCIA Y RECOLECCIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════
SEQUENCE_LENGTH  = 30      # Fotogramas por secuencia (eje temporal del LSTM)
NUM_SEQUENCES    = 60      # Meta recomendada de secuencias por clase (mínimo para entrenar: 30)
START_FOLDER     = 0       # Índice inicial de carpeta (útil para recolección incremental)
COLLECT_WAIT_MS  = 2000    # Pausa (ms) entre secuencias para que el usuario se prepare

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  UMBRALES DE PREDICCIÓN Y ESTABILIDAD
# ═══════════════════════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.85   # Confianza mínima para mostrar predicción [0.0–1.0]
STABILITY_FRAMES     = 5      # Fotogramas consecutivos con la misma predicción para confirmar
COOLDOWN_SECONDS     = 1.5    # Segundos de espera entre confirmaciones de la misma seña

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EXTRACCIÓN DE LANDMARKS
#
#     USE_FULL_FACE = True   →  1 662 características/frame (más preciso, requiere GPU)
#     USE_FULL_FACE = False  →    492 características/frame (más rápido, recomendado CPU)
# ═══════════════════════════════════════════════════════════════════════════════
USE_FULL_FACE = False   # ← Cambia a True si tienes GPU y quieres máxima precisión

# Índices clave de la malla facial de MediaPipe para LSM (78 puntos)
# Cubre: labios exteriores/interiores, cejas, ojos (corners + párpados), nariz, contorno
FACE_KEY_INDICES: list[int] = [
    # ── Labios: contorno exterior (20 pts) ──────────────────────────────────
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    # ── Labios: contorno interior (20 pts) ──────────────────────────────────
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    # ── Ceja izquierda (10 pts) ─────────────────────────────────────────────
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    # ── Ceja derecha (10 pts) ───────────────────────────────────────────────
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
    # ── Ojos — esquinas + párpados (8 pts) ──────────────────────────────────
    33, 133, 362, 263, 159, 145, 386, 374,
    # ── Nariz (5 pts) ───────────────────────────────────────────────────────
    1, 2, 4, 5, 6,
    # ── Contorno facial (5 pts) ─────────────────────────────────────────────
    152, 10, 234, 454, 93,
]   # Total: 78 puntos

# ── Dimensiones por región (NO MODIFICAR) ───────────────────────────────────
POSE_DIM       = 33 * 4                         # 132   (x, y, z, visibility)
FACE_FULL_DIM  = 468 * 3                        # 1404  (x, y, z)
FACE_KEY_DIM   = len(FACE_KEY_INDICES) * 3      # 234   (x, y, z)
LEFT_HAND_DIM  = 21 * 3                         # 63    (x, y, z)
RIGHT_HAND_DIM = 21 * 3                         # 63    (x, y, z)

FEATURE_DIM = (
    POSE_DIM + FACE_FULL_DIM + LEFT_HAND_DIM + RIGHT_HAND_DIM   # 1662 — full
    if USE_FULL_FACE else
    POSE_DIM + FACE_KEY_DIM  + LEFT_HAND_DIM + RIGHT_HAND_DIM   #  492 — reducido
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  RUTAS DEL PROYECTO
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(BASE_DIR, 'LSM_Dataset')
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
ASSETS_DIR      = os.path.join(BASE_DIR, 'assets')
SESSIONS_DIR    = os.path.join(BASE_DIR, 'sesiones')

MODEL_PATH      = os.path.join(MODELS_DIR, 'lsm_model.h5')
LABEL_MAP_PATH  = os.path.join(MODELS_DIR, 'label_map.json')
LOGO_LIA_PATH   = os.path.join(ASSETS_DIR, 'logo_lia.png')
LOGO_CECYT_PATH = os.path.join(ASSETS_DIR, 'logo_cecyt16.png')

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CÁMARA Y UI
# ═══════════════════════════════════════════════════════════════════════════════
CAMERA_INDEX = 0       # 0 = cámara predeterminada; cambiar si hay varias cámaras
FRAME_WIDTH  = 1280    # Resolución de captura
FRAME_HEIGHT = 720
TARGET_FPS   = 60      # FPS objetivo de la cámara
LOGO_HEIGHT  = 80      # Altura en píxeles de los logos en la UI
