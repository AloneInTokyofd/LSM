# Sistema de Reconocimiento de LSM Dinámica en Tiempo Real
### Laboratorio de Inteligencia Artificial (LIA) — CECyT 16, IPN

---

## 📁 Estructura del Proyecto

```
lsm_system/
│
├── main.py                ← 🚀 Reconocimiento en tiempo real (punto de entrada)
├── data_collection.py     ← 📷 Recolección de datos de entrenamiento
├── train.py               ← 🧠 Entrenamiento del modelo LSTM
├── model.py               ← 🏗️  Arquitectura de la red neuronal
├── config.py              ← ⚙️  Configuración global (editar aquí)
├── utils.py               ← 🛠️  Funciones auxiliares
├── requirements.txt       ← 📦 Dependencias de Python
│
├── assets/
│   ├── logo_lia.png       ← Logo LIA  (colocar aquí tu PNG con alfa)
│   └── logo_cecyt16.png   ← Logo CECyT 16 (colocar aquí tu PNG con alfa)
│
├── models/
│   ├── lsm_model.h5       ← Modelo entrenado (generado por train.py)
│   ├── label_map.json     ← Mapa índice→seña  (generado por train.py)
│   ├── training_history.png
│   └── confusion_matrix.png
│
├── LSM_Dataset/           ← Dataset de secuencias (generado por data_collection.py)
│   ├── hola/
│   │   ├── 0/
│   │   │   ├── frame_000.npy   ← Keypoints del frame 0  (shape: 492 o 1662,)
│   │   │   ├── frame_001.npy
│   │   │   └── ...  (30 archivos por secuencia)
│   │   ├── 1/
│   │   └── ...  (30 carpetas por seña)
│   ├── gracias/
│   └── ...
│
└── sesiones/
    └── sesion_20250101_120000.txt  ← Oraciones guardadas con 'S'
```

---

## ⚙️ Configuración Inicial (`config.py`)

Antes de ejecutar cualquier script, abre `config.py` y ajusta:

```python
# 1. Señas que vas a reconocer
ACTIONS = np.array(['hola', 'gracias', 'por_favor', ...])

# 2. Extracción de landmarks
USE_FULL_FACE = False   # False = 492 features (CPU)  |  True = 1662 features (GPU)

# 3. Umbrales de predicción
CONFIDENCE_THRESHOLD = 0.85   # Subir si hay muchos falsos positivos
STABILITY_FRAMES     = 5      # Subir para exigir más consistencia

# 4. Dataset
SEQUENCE_LENGTH = 30     # Fotogramas por secuencia
NUM_SEQUENCES   = 30     # Secuencias por clase (mínimo recomendado: 30)
```

---

## 🚀 Flujo de Trabajo Completo

### Paso 0 — Instalación
```bash
# Crear entorno virtual (recomendado)
python -m venv venv_lsm
source venv_lsm/bin/activate      # Linux/Mac
venv_lsm\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 1 — Colocar los Logos
Copia tus archivos PNG en la carpeta `assets/`:
```
assets/logo_lia.png
assets/logo_cecyt16.png
```
> Los PNGs deben tener **canal alfa** (RGBA) para la transparencia.
> Si no tienes logos, el sistema funciona sin ellos (solo muestra advertencia).

### Paso 2 — Recolección de Datos
```bash
python data_collection.py
```

**Controles durante la recolección:**
| Tecla | Acción |
|-------|--------|
| `SPACE` | Iniciar grabación de la siguiente secuencia |
| `R` | Repetir la secuencia actual (si algo salió mal) |
| `Q` / `ESC` | Abortar y guardar lo recolectado |

**Consejos para buena recolección:**
- Iluminación uniforme y fondo sin movimiento.
- Realizar la seña de forma natural y completa dentro de los 30 frames.
- Variar ligeramente la posición y velocidad en cada secuencia.
- Mínimo 30 secuencias por seña; 60+ para mayor robustez.

### Paso 3 — Entrenamiento
```bash
python train.py
```

El script generará automáticamente:
- `models/lsm_model.h5` — Modelo LSTM entrenado.
- `models/label_map.json` — Mapa de etiquetas.
- `models/training_history.png` — Gráfica de accuracy/loss.
- `models/confusion_matrix.png` — Matriz de confusión del test set.

**Parámetros ajustables en `train.py`:**
```python
BATCH_SIZE           = 16      # Reducir si hay poca VRAM
EPOCHS               = 200     # EarlyStopping detendrá antes si converge
USE_ATTENTION_MODEL  = False   # True para el modelo con atención (más lento)
USE_CLASS_WEIGHTS    = True    # True si tienes clases desbalanceadas
```

### Paso 4 — Reconocimiento en Tiempo Real
```bash
python main.py
```

**Controles durante el reconocimiento:**
| Tecla | Acción |
|-------|--------|
| `Q` / `ESC` | Salir del sistema |
| `R` | Resetear buffer y estado de predicción |
| `C` | Limpiar la oración acumulada en pantalla |
| `S` | Guardar la oración actual en `sesiones/` |

---

## 🧠 Arquitectura del Modelo

```
Input (batch, 30, 492)
  │
  ├─ TimeDistributed(Dense 256)    ← Proyección: reduce features por frame
  │  BatchNorm + Dropout(0.20)
  │
  ├─ BiLSTM(256, →→←←)            ← Contexto temporal bidireccional
  │  BatchNorm + Dropout(0.30)
  │
  ├─ BiLSTM(128, →→←←)            ← Refinamiento temporal
  │  BatchNorm + Dropout(0.30)
  │
  ├─ LSTM(64)                      ← Colapsa dimensión temporal
  │  BatchNorm + Dropout(0.30)
  │
  ├─ Dense(256, relu)
  │  BatchNorm + Dropout(0.40)
  │
  ├─ Dense(128, relu)
  │
  └─ Dense(num_clases, softmax)    ← Probabilidad por seña
```

---

## 🔍 Lógica de Predicción (Doble Umbral)

```
Por cada frame:
  1. Extraer 492 keypoints (pose + cara key + manos)
  2. Añadir al buffer circular de 30 frames

Cuando el buffer está lleno:
  3. Inferencia LSTM → probabilidades por clase

  4. ¿Confianza ≥ 0.85?
       NO  → Resetear estabilidad. Esperar.
       SÍ  → Continuar...

  5. ¿Es la misma seña que el frame anterior?
       NO  → Reiniciar contador de estabilidad a 1.
       SÍ  → Incrementar contador.

  6. ¿Contador ≥ 5 frames consecutivos?
       NO  → Mostrar indicador tentativo (gris). Esperar.
       SÍ  → ¡Confirmar predicción! (verde + panel inferior)
              Añadir a oración. Activar cooldown de 1.5s.
```

---

## 🔧 Dimensiones de Keypoints

| Región | Puntos | Coords | Valores |
|--------|--------|--------|---------|
| Pose (esqueleto) | 33 | x,y,z,vis | 132 |
| Cara (key points) | 78 | x,y,z | 234 |
| Mano izquierda | 21 | x,y,z | 63 |
| Mano derecha | 21 | x,y,z | 63 |
| **Total (reducido)** | | | **492** |
| Cara (completa) | 468 | x,y,z | 1404 |
| **Total (completo)** | | | **1662** |

> `USE_FULL_FACE = False` (recomendado para CPU)
> `USE_FULL_FACE = True`  (recomendado con GPU, mayor precisión facial)

---

## ❗ Solución de Problemas Frecuentes

**El modelo no detecta nada:**
- Verifica que `USE_FULL_FACE` en `config.py` sea igual al valor usado al recolectar datos.
- Comprueba que haya suficiente iluminación.
- Prueba ajustar `CONFIDENCE_THRESHOLD` a 0.70 para ser menos restrictivo.

**Muchos falsos positivos / "parpadeos":**
- Aumenta `CONFIDENCE_THRESHOLD` a 0.90.
- Aumenta `STABILITY_FRAMES` a 7 o 10.
- Aumenta `COOLDOWN_SECONDS` a 2.0 o 3.0.

**Baja precisión en test:**
- Aumenta `NUM_SEQUENCES` a 60+ secuencias por clase.
- Verifica que todas las señas se graben con condiciones similares.
- Usa `USE_ATTENTION_MODEL = True` si tienes GPU.

**Error de dimensión (`FEATURE_DIM`):**
- Asegúrate de que `USE_FULL_FACE` sea el mismo valor en `config.py`
  al momento de recolectar datos, entrenar y ejecutar `main.py`.
- Si cambias este valor, debes recolectar datos nuevamente.

**MediaPipe no detecta las manos:**
- Asegúrate de estar usando `mediapipe==0.10.14` (última versión con Holistic estable).
- Verifica que no haya objetos brillantes o similares a manos en el fondo.
