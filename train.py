"""
train.py — Pipeline de entrenamiento del modelo LSTM para LSM dinámica.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Laboratorio de Inteligencia Artificial (LIA) - CECyT 16, IPN.

PREREQUISITO
─────────────
  Ejecutar data_collection.py primero para generar LSM_Dataset/.

PIPELINE
─────────
  1. Carga secuencias desde LSM_Dataset/ → arrays NumPy.
  2. Codificación one-hot de etiquetas.
  3. División estratificada: train / validación / test.
  4. Calcula pesos de clase (compensa datasets desbalanceados).
  5. Construye modelo LSTM (o Atención si USE_ATTENTION_MODEL=True).
  6. Entrena con callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
  7. Evalúa en conjunto de test y reporta métricas finales.
  8. Guarda modelo .h5 y label_map.json.
  9. Genera gráfica de accuracy/loss.

EJECUCIÓN
──────────
  python train.py
"""
from __future__ import annotations
import os
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from config import (
    ACTIONS, DATA_PATH, MODEL_PATH, LABEL_MAP_PATH, MODELS_DIR,
    SEQUENCE_LENGTH, NUM_SEQUENCES, START_FOLDER, FEATURE_DIM,
)
from model import build_lstm_model, build_attention_model


# ══════════════════════════════════════════════════════════════════════════════
# HIPERPARÁMETROS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE           = 16      # Tamaño del mini-batch
EPOCHS               = 200     # Épocas máximas (EarlyStopping detendrá antes)
VALIDATION_SPLIT     = 0.15    # 15 % para validación
TEST_SPLIT           = 0.15    # 15 % para evaluación final
USE_ATTENTION_MODEL  = False   # True → modelo con atención (recomienda GPU)
USE_CLASS_WEIGHTS    = True    # Balancea clases desiguales automáticamente
RANDOM_SEED          = 42


# ══════════════════════════════════════════════════════════════════════════════
# A.  CARGA DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset() -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """
    Recorre LSM_Dataset/ y carga todas las secuencias como arrays NumPy.

    Manejo de errores:
      · Si falta un frame .npy, se rellena con ceros (sin detección).
      · Si la dimensión de un frame no coincide con FEATURE_DIM, se omite
        la secuencia y se emite un aviso.
      · Señas sin secuencias válidas generan una advertencia, no un crash.

    Returns:
        X         : float32 array (N, SEQUENCE_LENGTH, FEATURE_DIM)
        y         : int32   array (N,) — etiquetas enteras
        label_map : {int: str}  — índice → nombre de seña
    """
    sequences:  list[np.ndarray] = []
    labels:     list[int]        = []
    label_map:  dict[int, str]   = {}
    stats:      dict[str, int]   = {}

    print('\n[INFO] Cargando dataset...')
    print(f'       FEATURE_DIM     = {FEATURE_DIM}')
    print(f'       SEQUENCE_LENGTH = {SEQUENCE_LENGTH}')
    print(f'       Señas           = {list(ACTIONS)}\n')

    for label_idx, action in enumerate(ACTIONS):
        label_map[label_idx] = action
        loaded = 0
        skipped = 0

        for seq_idx in range(START_FOLDER, START_FOLDER + NUM_SEQUENCES):
            seq_path = os.path.join(DATA_PATH, action, str(seq_idx))
            if not os.path.isdir(seq_path):
                skipped += 1
                continue

            window: list[np.ndarray] = []
            valid = True

            for frame_num in range(SEQUENCE_LENGTH):
                npy = os.path.join(seq_path, f'frame_{frame_num:03d}.npy')
                if os.path.exists(npy):
                    kp = np.load(npy).astype(np.float32)
                    if kp.shape[0] != FEATURE_DIM:
                        print(f'  [SKIP] {npy}: dim={kp.shape[0]} ≠ {FEATURE_DIM}')
                        valid = False
                        skipped += 1
                        break
                    window.append(kp)
                else:
                    # Frame faltante → cero (sin detección en ese instante)
                    window.append(np.zeros(FEATURE_DIM, dtype=np.float32))

            if valid and len(window) == SEQUENCE_LENGTH:
                sequences.append(np.stack(window, axis=0))
                labels.append(label_idx)
                loaded += 1
            elif valid:
                skipped += 1

        stats[action] = loaded
        status = '✓' if loaded > 0 else '✗'
        skip_txt = f'  ({skipped} omitidas)' if skipped else ''
        print(f'  {status}  {action:<20}  {loaded:>3} secuencias{skip_txt}')

    if not sequences:
        raise RuntimeError(
            '\n[ERROR] No se encontró ninguna secuencia válida.\n'
            '        Asegúrate de haber ejecutado data_collection.py primero\n'
            f'        y que USE_FULL_FACE en config.py coincida con los datos '
            f'recolectados (FEATURE_DIM={FEATURE_DIM}).'
        )

    X = np.stack(sequences, axis=0).astype(np.float32)   # (N, T, F)
    y = np.array(labels, dtype=np.int32)

    print(f'\n[INFO] Dataset total: X={X.shape}  y={y.shape}')
    dist = {ACTIONS[i]: int(np.sum(y == i)) for i in range(len(ACTIONS))}
    print(f'[INFO] Distribución : {dist}')
    return X, y, label_map


# ══════════════════════════════════════════════════════════════════════════════
# B.  VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history: tf.keras.callbacks.History,
                          save_dir: str = MODELS_DIR) -> None:
    """Gráfica de accuracy y loss para train/validación."""
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    for ax_idx, (metric, title) in enumerate([
        ('accuracy', 'Precisión del Modelo'),
        ('loss',     'Pérdida del Modelo'),
    ]):
        ax = fig.add_subplot(gs[ax_idx])
        ax.plot(history.history[metric],
                label='Entrenamiento', color='#4C72B0', lw=2)
        ax.plot(history.history[f'val_{metric}'],
                label='Validación',    color='#DD8452', lw=2, ls='--')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel(title.split()[0])
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric == 'accuracy':
            ax.set_ylim(0, 1.05)

    fig.suptitle('LSM — Historial de Entrenamiento LSTM',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'[INFO] Gráfica de entrenamiento → {path}')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          label_map: dict[int, str],
                          save_dir: str = MODELS_DIR) -> None:
    """Matriz de confusión normalizada para el conjunto de test."""
    labels = [label_map[i] for i in sorted(label_map.keys())]
    cm     = confusion_matrix(y_true, y_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels))))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Real',       fontsize=12)
    ax.set_title('Matriz de Confusión (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'[INFO] Matriz de confusión → {path}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# C.  CALLBACK PERSONALIZADO — LOG DE ÉPOCA
# ══════════════════════════════════════════════════════════════════════════════

class EpochLogger(Callback):
    """Imprime un resumen compacto al final de cada época."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        tr_acc  = logs.get('accuracy',     0) * 100
        val_acc = logs.get('val_accuracy', 0) * 100
        tr_loss = logs.get('loss',         0)
        val_loss= logs.get('val_loss',     0)
        lr      = logs.get('lr', 0)

        if (epoch + 1) % 10 == 0:
            print(f'  Época {epoch + 1:>4}  │  '
                  f'Acc: {tr_acc:5.1f}% / Val: {val_acc:5.1f}%  │  '
                  f'Loss: {tr_loss:.4f} / Val: {val_loss:.4f}  │  '
                  f'LR: {lr:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# D.  PIPELINE DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train() -> None:
    """Pipeline completo: cargar datos → entrenar → evaluar → guardar."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.time()

    # ── 1. Cargar datos ──────────────────────────────────────────
    X, y, label_map = load_dataset()
    num_classes     = len(label_map)

    # ── 2. One-hot encoding ──────────────────────────────────────
    lb       = LabelBinarizer()
    y_onehot = lb.fit_transform(y).astype(np.float32)
    if y_onehot.shape[1] == 1:          # Edge case: dataset binario
        y_onehot = np.hstack([1 - y_onehot, y_onehot])

    # ── 3. Split: train / val / test ─────────────────────────────
    X_tv, X_test, y_tv, y_test, y_tv_int, y_test_int = train_test_split(
        X, y_onehot, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    val_frac = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_frac,
        random_state=RANDOM_SEED,
    )

    print(f'\n[INFO] Split → Train: {len(X_train)}  '
          f'Val: {len(X_val)}  Test: {len(X_test)}')

    # ── 4. Pesos de clase (para datasets desbalanceados) ─────────
    class_weights = None
    if USE_CLASS_WEIGHTS:
        y_train_int = np.argmax(y_train, axis=1)
        weights     = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),
            y=y_train_int,
        )
        class_weights = dict(enumerate(weights))
        print('[INFO] Pesos de clase:')
        for k, v in class_weights.items():
            print(f'       {label_map[k]:<20} → {v:.3f}')

    # ── 5. Construir modelo ───────────────────────────────────────
    builder = build_attention_model if USE_ATTENTION_MODEL else build_lstm_model
    model   = builder(num_classes)
    model.summary()
    print(f'\n[INFO] {"Attention" if USE_ATTENTION_MODEL else "BiLSTM"} model  '
          f'| FEATURE_DIM={FEATURE_DIM}  | SEQUENCE_LENGTH={SEQUENCE_LENGTH}')

    # ── 6. Callbacks ──────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy', mode='max',
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss', patience=35,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=12,
            min_lr=1e-7, verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(MODELS_DIR, 'logs'),
            histogram_freq=0,
        ),
        EpochLogger(),
    ]

    # ── 7. Entrenamiento ──────────────────────────────────────────
    print(f'\n[INFO] Entrenando — máx. {EPOCHS} épocas, batch={BATCH_SIZE}...\n')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,    # EpochLogger se encarga del log
    )

    elapsed = time.time() - t0
    print(f'\n[INFO] Tiempo de entrenamiento: {elapsed / 60:.1f} minutos')

    # ── 8. Evaluación en test ─────────────────────────────────────
    print('\n[INFO] Evaluando en conjunto de prueba (test set)...')
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'  ► Test Loss     : {test_loss:.4f}')
    print(f'  ► Test Accuracy : {test_acc:.4f}  ({test_acc:.1%})')

    # Reporte detallado por clase
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_int   = np.argmax(y_pred_probs, axis=1)
    class_names  = [label_map[i] for i in range(num_classes)]
    print('\n[INFO] Reporte por clase:\n')
    print(classification_report(y_test_int, y_pred_int,
                                 target_names=class_names, digits=3))

    # ── 9. Guardar label map ──────────────────────────────────────
    with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f'[INFO] Label map → {LABEL_MAP_PATH}')
    print(f'[INFO] Modelo    → {MODEL_PATH}')

    # ── 10. Visualizaciones ───────────────────────────────────────
    plot_training_history(history)
    try:
        plot_confusion_matrix(y_test_int, y_pred_int, label_map)
    except ImportError:
        print('[WARN] seaborn no instalado — omitiendo matriz de confusión.')

    print(f'\n[✓] Entrenamiento finalizado con éxito.')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Configurar GPU si está disponible
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'[INFO] GPU detectada: {[g.name for g in gpus]}')
    else:
        print('[INFO] GPU no detectada — usando CPU (puede ser lento).')

    train()
