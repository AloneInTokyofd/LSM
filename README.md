# Sistema de Reconocimiento de LSM Dinámica en Tiempo Real

### Modelo web v3: forma de la mano y comparación antes de activarlo

Los nuevos entrenamientos añaden 128 características por fotograma: coordenadas
de los dedos relativas a su muñeca, escaladas por el tamaño de la palma, y dos
indicadores de presencia. Se conserva la orientación de las manos y el contexto
temporal; no se refleja ni gira el gesto. Esto permite representar la forma
de los dedos con menor dependencia de dónde está la persona en el encuadre.

También se guardan referencias de forma tomadas exclusivamente del entrenamiento.
Su tolerancia se comprueba con validación. Si una predicción se aleja de esas
referencias, la página la muestra como dudosa y explica el rechazo, aunque la
red dé una confianza alta. No demuestra reconocimiento de todas las señas
desconocidas: hay que comprobarlo con nuevos ejemplos reales.

Antes de activar una versión con el mismo vocabulario, se comparan ambas sobre
las mismas secuencias de validación: aciertos, F1, precisión de predicciones
aceptadas y cobertura. Si cualquiera empeora, se conserva el modelo anterior.
La prueba reservada se usa para el informe, nunca para decidir qué versión activar.
Los modelos v1/v2 y los ejemplos ya guardados permanecen compatibles.

### Requisitos actuales y mejoras de entrenamiento

Puedes grabar tandas desde **5 tomas** y acumularlas. Para **entrenar y probar**
se requieren **30 ejemplos completos por palabra o letra** y al menos
**dos señas distintas**. Se recomiendan **30 a 60 ejemplos por palabra o letra**.
Las tandas consecutivas admiten de 5 a 60 tomas (30 por defecto). Puedes superar 60 ejemplos
acumulando más tandas. Incluye distintas personas, días, velocidades y posiciones;
repetir tomas casi idénticas aporta menos variedad que recoger ejemplos nuevos.

Los nuevos modelos web usan una BiLSTM más compacta para limitar la memorización,
coordenadas relativas al torso y escala de hombros, y normalización por característica
calculada solo con el conjunto de entrenamiento. La misma transformación se aplica
al reconocer. Los archivos de grabaciones originales no se transforman ni se borran.
Solo el conjunto de entrenamiento recibe dos variaciones leves de escala y ruido,
sin reflejar las manos ni invertir los movimientos. Validación y prueba quedan
reservadas y sin aumento de datos. Las variantes artificiales no cuentan para
cumplir el mínimo de grabaciones reales.

El mejor punto del entrenamiento se elige por pérdida de validación. Se ajusta
un umbral de confianza con validación y se exige margen entre las dos primeras
clases. En directo también se promedian cinco predicciones y se mantiene la
comprobación de estabilidad temporal. La captura requiere manos visibles en
al menos el 80 % de los fotogramas.

La página muestra precisión, recuperación y F1 por seña, además de cobertura y
aciertos de predicciones aceptadas en la prueba reservada. Si el filtro rechaza
todo, la precisión aceptada figura sin valor, nunca como 100 %. Estos cambios
son medidas para mejorar la generalización; no demuestran una mejora hasta
reentrenar y medir con datos reales. Los modelos anteriores siguen siendo
compatibles, y necesitan reentrenarse para incorporar el nuevo preprocesamiento.

### Grabación consecutiva y guía rápida de la página

En **Capturar y entrenar**, activa la cámara y abre **Grabar ejemplos
consecutivos**. Elige de 5 a 60 tomas (recomendado: 30 a 60) y una pausa de 1 a 10 segundos (por defecto,
30 tomas con 3 segundos de preparación). Pulsa **Iniciar grabación consecutiva**
una vez. Cada cuenta atrás da paso a otra secuencia completa, sin volver a
pulsar “Grabar una seña”. Todas las tomas quedan pendientes en la lengua actual.

Al terminar, usa **Grabaciones pendientes** para nombrar y guardar cada una. Para
repeticiones de una misma seña, usa el mismo nombre. **Detener tanda**, apagar
la cámara, cambiar de lengua o sección u ocultar la página detiene la captura;
las tomas completas permanecen en disco y la captura parcial se descarta.
Los errores de detección o conexión detienen la tanda para que puedas corregir
el encuadre. Se admiten hasta 100 tomas pendientes por lengua antes de nombrar
y guardar algunas. Se guardan puntos para entrenamiento, sin archivos de video
ni audio.

La sección **Guía rápida · ¿Cómo uso esta app?**, debajo del encabezado, explica
el recorrido completo: elegir lengua, grabar, nombrar y guardar, entrenar y probar.
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
NUM_SEQUENCES   = 60     # Meta por clase (mínimo para entrenar: 30)
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

#### Cargar el modelo para detectar señas

El proyecto no incluye un modelo preentrenado. Primero recolecta ejemplos para
cada seña configurada en `ACTIONS` ejecutando `data_collection.py`. Después,
entrena el modelo:

```bash
.venv/bin/python train.py
```

Al terminar, `train.py` guarda `models/lsm_model.h5` y
`models/label_map.json`. `main.py` los busca automáticamente en esas rutas al
arrancar; no necesitas cargar el archivo manualmente en el código:

```bash
.venv/bin/python main.py
```

Mantén `SEQUENCE_LENGTH` y `USE_FULL_FACE` iguales durante recolección,
entrenamiento y detección. Si cambias la orientación, las señas o esas
características, vuelve a recolectar y entrenar el modelo. Si falta el modelo,
`main.py` avisa y entra en modo demo, sin detecciones.

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


## Página local: siete lenguas de señas

Inicia la página usando el entorno del proyecto:

```bash
.venv/bin/python web_app.py
```

Abre http://127.0.0.1:8765 en tu navegador y permite la cámara cuando pulses
**Activar cámara**. La página funciona localmente; no necesita Flask, servicios
externos ni claves de API. Para cambiar el puerto usa `--port 8766`.

Los siete perfiles son **LSM (México), DGS (Alemania), ASL (Estados Unidos),
JSL (Japón), CSL (China), LSF (Francia) y Libras (Brasil)**. Son lenguas de
señas distintas. Los nombres español, alemán, inglés, japonés, chino mandarín,
francés y portugués corresponden al texto de las etiquetas de salida. ASL no
representa todas las lenguas de señas de países anglófonos; Libras es la variante
brasileña seleccionada para este proyecto.

### Capturar, entrenar y probar

1. Elige una lengua en la barra lateral.
2. En **Capturar y entrenar**, activa la cámara y pulsa **Grabar una seña**.
   Graba la seña auténtica de esa lengua. Al terminar, escribe su nombre en
   **¿Qué significa esta seña?** y pulsa **Guardar e integrar**. Si el nombre
   ya existe, suma otro ejemplo; si es nuevo, se añade al vocabulario de esa
   lengua automáticamente. La página hace una cuenta de tres
   segundos y recoge 30 fotogramas a un objetivo de 10 por segundo. La velocidad
   efectiva depende de tu computadora. No uses las señas de LSM como ejemplos
   de ASL u otra lengua.
3. Completa al menos **30 ejemplos por palabra o letra para dos señas distintas** (recomendado: **30–60 por palabra o letra**). Toda
   seña que empezaste necesita ese mínimo antes de entrenar. El mínimo permite
   dividir los datos; no garantiza precisión suficiente. Incluye distintas
   personas, iluminación y sesiones para obtener mayor variedad.
4. Pulsa **Entrenar modelo**. El progreso se muestra en la página. Se reserva
   aproximadamente el 15 % para validación y otro 15 % para prueba, con un
   mínimo de dos ejemplos por clase en cada conjunto. Las secuencias se separan
   completas; no se mezclan sus fotogramas entre conjuntos.
5. Al terminar, el modelo se activa solo para esa lengua. En **Probar modelo**,
   realiza una seña del vocabulario entrenado. Puedes elegir la seña esperada
   para contar aciertos y errores de la sesión, escuchar el resultado si tu
   navegador dispone de la voz correspondiente y descargar las confirmaciones.
6. Baja las manos unos instantes para repetir una misma seña. Al cambiar de
   lengua o sección se apaga la cámara y se descarta la captura parcial.

La página no incluye siete modelos preentrenados ni interpreta conversaciones
libres. Las etiquetas escritas son un vocabulario inicial, no demostraciones
de los gestos. Cada perfil sin modelo lo indica explícitamente. Una precisión
alta con el conjunto reservado de tus propios datos no demuestra precisión
con nuevos usuarios, variantes regionales o conversaciones cotidianas.

### Archivos de la página

- `sign_languages.py`: perfiles, vocabulario y configuración de captura.
- `web_app.py`: servidor local, captura desde navegador y reconocimiento.
- `web_dataset.py`: validación de secuencias completas y división por clase.
- `web_train.py`: entrenamiento independiente por lengua y progreso.
- `static/`: interfaz de la página, sin dependencias remotas.
- `web_data/datasets/<lengua>/<seña>/<id>.npz`: coordenadas de las secuencias.
  No se guardan imágenes ni videos.
- `web_data/jobs/<id>.json` y `.log`: progreso y errores del entrenamiento.
- `models/<lengua>/<versión>/model.keras`: pesos entrenados.
- `models/<lengua>/<versión>/metadata.json`: etiquetas, configuración y métricas.
- `models/<lengua>/current.json`: versión activa; se publica solo después de
  guardar el modelo y sus metadatos completos.

El modelo de escritorio `models/lsm_model.h5` y `LSM_Dataset/` se conservan.
La página no activa ese modelo automáticamente: usa su propio ritmo de captura
y requiere metadatos compatibles. Recolecta ejemplos en la página para entrenar
un modelo web. El entrenador de escritorio también omite ahora las secuencias
incompletas, en lugar de inventar ejemplos rellenándolos con ceros.

### Recursos para ampliar las lenguas

El [compendio de la Universidad de Hamburgo](https://www.sign-lang.uni-hamburg.de/lr/compendium/language/index.html)
permite buscar datasets y diccionarios por lengua. [OpenHands](https://aclanthology.org/2022.acl-long.150/)
presenta modelos basados en pose y transferencia entre lenguas. Estos recursos
no constituyen un paquete de siete modelos compatibles con esta arquitectura:
su uso requiere revisar la licencia y adaptar etiquetas, puntos y entrenamiento.


### Grabar primero y poner nombre después

El flujo de la página es **elegir lengua → grabar → nombrar → guardar → entrenar**.
Los nombres pueden escribirse en español, alemán, inglés, japonés, chino,
francés o portugués, hasta 80 caracteres. El catálogo por lengua admite nuevas
señas sin editar `ACTIONS` ni los archivos Python. Escribir de nuevo el mismo
nombre (sin distinguir mayúsculas o espacios extra) añade otro ejemplo a esa
clase dentro de esa lengua.

Las capturas completas pendientes se guardan en `web_data/drafts/<lengua>/` y
reaparecen al recargar o volver a seleccionar esa lengua. La captura incompleta
se descarta al apagar la cámara. **Descartar y repetir** mueve una captura
completa a la subcarpeta `discarded/`, fuera del entrenamiento.

Al pulsar **Guardar e integrar**, el nombre se registra en
`web_data/vocabulary/<lengua>.json` y el ejemplo pasa a
`web_data/datasets/<lengua>/<id de seña>/`. Los nombres visibles nunca se usan
como rutas de archivos. El vocabulario, el contador de ejemplos y el siguiente
entrenamiento leen automáticamente ese catálogo. Reintentar el mismo guardado
no duplica el ejemplo.

Una seña guardada aparece como **Por entrenar** hasta formar parte de un modelo.
Guardar un ejemplo no actualiza los pesos: reúne el mínimo indicado para las
clases que empezaste y pulsa **Entrenar modelo**. Al terminar, la nueva versión
se activa para esa lengua y la seña aparece en **Probar modelo**.


### Abecedarios y caracteres por lengua

En **Capturar y entrenar → Abecedario y caracteres** puedes añadir todas las
entradas básicas de una lengua o las bases de las siete lenguas con un botón.
El catálogo se guarda por lengua y reutiliza las letras existentes, sin duplicar
ni borrar sus ejemplos. Las letras sin grabaciones no bloquean el entrenamiento.

- LSM: A–Z y Ñ (27 letras); vocales acentuadas y Ü como variantes opcionales.
- DGS: A–Z, Ä, Ö, Ü y ß (30 entradas escritas).
- ASL, LSF y Libras: A–Z (26 letras). Francés y portugués ofrecen también
  caracteres acentuados y variantes escritas opcionales.
- JSL: 46 kana básicos; variantes sonoras, kana pequeños y marca de vocal larga
  como catálogo adicional. No representa todos los kanji.
- CSL: A–Z, ZH, CH, SH y NG para recolectar deletreo basado en pinyin;
  Ü como variante escrita opcional. No representa todos los caracteres chinos.

Pulsa una letra para seleccionarla, activa la cámara y graba su seña. Al terminar,
se propondrá esa letra como nombre: confírmala con **Guardar e integrar**. Puedes
repetir el proceso para sumar ejemplos o volver a grabar señas libres. La cuadrícula
muestra ejemplos por letra y si ya pertenece al modelo activo. Entrenar sigue
siendo necesario para reconocer las letras; las variantes escritas no implican
que cada una tenga una configuración manual distinta.

Referencias para el catálogo y las convenciones de deletreo:
[SEP, abecedario en LSM](https://nuevaescuelamexicana.sep.gob.mx/contenido/recurso/40967/),
[Deutsches Taubblindenwerk, Fingeralphabet](https://www.taubblindenwerk.de/haeufig-gestellte-fragen/fingeralphabet/),
[Ciudad de Nara, 指文字](https://www.city.nara.lg.jp/soshiki/34/184979.html) y
[Ministerio de Educación de China, 汉语手指字母方案](https://www.moe.gov.cn/jyb_sjzl/ziliao/A19/201404/t20140422_167548.html).

### Cámaras y cambios moderados de encuadre

La sección **Cámara**, encima de la vista previa, lista las cámaras que el navegador
puede detectar. Usa **Detectar cámaras** para actualizarla. Tras autorizar el acceso,
aparecen los nombres disponibles. Puedes elegir una cámara integrada o externa;
si está encendida, cambiar de cámara detiene la toma parcial y la tanda, conserva
las tomas completas pendientes y reinicia la vista con la cámara seleccionada.
Desconectar la cámara activa detiene la captura. Los permisos del navegador y de
macOS pueden limitar qué dispositivos aparecen.

Los nuevos entrenamientos amplían solo el conjunto de entrenamiento con
inclinaciones XY de 4–10 grados y escalas horizontales de 0,95–1,05, constantes
durante cada toma. Después se calculan las características de cada variante,
incluida la forma de las manos. El filtro de forma utiliza también estas variantes.
No se reflejan manos ni se invierte el tiempo. Las estadísticas de normalización,
validación y prueba siguen usando ejemplos originales; las variantes no cuentan
para el mínimo de 30 grabaciones reales por palabra o letra. La comparación con
el modelo anterior sigue evitando activar uno con peores métricas de validación.

**Reentrena** para aplicar estas mejoras. No garantizan el mismo rendimiento desde
cualquier ángulo: no reconstruyen manos ocultas ni simulan correctamente giros 3D.
Graba ejemplos reales desde varios ángulos moderados, distancias y sesiones;
comprueba después con tomas nuevas desde cada posición. Algunas orientaciones
cambian el significado de la seña y deben conservarse.
