"""
model.py — Arquitectura de la red neuronal para reconocimiento de LSM dinámica.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Laboratorio de Inteligencia Artificial (LIA) - CECyT 16, IPN.

MODELOS DISPONIBLES
───────────────────
  1. build_lstm_model()       BiLSTM con capa de proyección temporal.
                              Recomendado para empezar. Funciona en CPU y GPU.

  2. build_attention_model()  BiLSTM + Atención Multi-Cabeza (Transformer-lite).
                              Más preciso para señas con movimientos complejos.
                              Requiere GPU para entrenamiento en tiempo razonable.

DECISIÓN DE DISEÑO: Capa de Proyección Temporal
───────────────────────────────────────────────
El vector de características por frame (FEATURE_DIM) puede ser grande
(492 o 1 662 valores). Alimentar este vector directamente a las celdas LSTM
genera millones de parámetros innecesarios en la primera capa recurrente.

Solución: TimeDistributed(Dense(256)) proyecta cada frame de
(FEATURE_DIM,) → (256,) antes de entrar al LSTM, reduciendo parámetros
sin pérdida significativa de información.

EJECUCIÓN DIRECTA
─────────────────
  python model.py   →  imprime resumen de ambos modelos y sus shapes.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Bidirectional, TimeDistributed, GlobalAveragePooling1D,
    MultiHeadAttention, Add, LayerNormalization,
)
from tensorflow.keras.regularizers import l2

from config import SEQUENCE_LENGTH, FEATURE_DIM, ACTIONS


def build_web_model(num_classes: int, learning_rate: float = 5e-4, input_dim: int = FEATURE_DIM) -> tf.keras.Model:
    """BiLSTM más compacto para datasets personales con pocas decenas de ejemplos."""
    model = Sequential(name='Sign_Web_Compact', layers=[
        Input(shape=(SEQUENCE_LENGTH, input_dim)),
        TimeDistributed(Dense(96, activation='relu', kernel_regularizer=l2(1e-4))),
        LayerNormalization(),
        Dropout(.2),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))),
        Dropout(.3),
        Bidirectional(LSTM(32, kernel_regularizer=l2(1e-4))),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(.4),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 1: BiLSTM con Proyección Temporal
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_model(num_classes: int = len(ACTIONS),
                     learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    LSTM Bidireccional con capa de proyección temporal.

    Arquitectura detallada
    ──────────────────────
    Input  (batch, SEQUENCE_LENGTH, FEATURE_DIM)
      │
      ├─ [Proyección] TimeDistributed(Dense 256, relu)
      │   Reduce cada frame: FEATURE_DIM → 256
      │   BatchNorm + Dropout(0.20)
      │
      ├─ [BiLSTM-1] Bidirectional(LSTM 256, return_sequences=True)
      │   Captura contexto temporal forward + backward.
      │   Salida: (batch, T, 512)
      │   BatchNorm + Dropout(0.30)
      │
      ├─ [BiLSTM-2] Bidirectional(LSTM 128, return_sequences=True)
      │   Refina representación temporal.
      │   Salida: (batch, T, 256)
      │   BatchNorm + Dropout(0.30)
      │
      ├─ [LSTM-Final] LSTM(64, return_sequences=False)
      │   Colapsa la dimensión temporal → un vector por secuencia.
      │   Salida: (batch, 64)
      │   BatchNorm + Dropout(0.30)
      │
      ├─ [Dense-1] Dense(256, relu) + L2(1e-4)
      │   BatchNorm + Dropout(0.40)
      │
      ├─ [Dense-2] Dense(128, relu)
      │
      └─ [Salida] Dense(num_classes, softmax)

    Args:
        num_classes:   Número de señas (clases) a reconocer.
        learning_rate: Tasa de aprendizaje inicial del optimizador Adam.
    Returns:
        Modelo Keras compilado, listo para .fit().
    """
    model = Sequential(name='LSM_BiLSTM', layers=[
        Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name='input_seq'),

        # ── Proyección: FEATURE_DIM → 256 (por cada frame) ───────
        TimeDistributed(
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            name='time_projection'
        ),
        BatchNormalization(name='proj_bn'),
        Dropout(0.20, name='proj_drop'),

        # ── BiLSTM 1 — contexto bidireccional completo ────────────
        Bidirectional(
            LSTM(256,
                 return_sequences=True,
                 dropout=0.20,
                 recurrent_dropout=0.0,     # 0 para compatibilidad con cuDNN (GPU)
                 kernel_regularizer=l2(1e-4)),
            name='bilstm_1'
        ),
        BatchNormalization(name='bilstm1_bn'),
        Dropout(0.30, name='bilstm1_drop'),

        # ── BiLSTM 2 — refinamiento temporal ─────────────────────
        Bidirectional(
            LSTM(128,
                 return_sequences=True,
                 dropout=0.20,
                 kernel_regularizer=l2(1e-4)),
            name='bilstm_2'
        ),
        BatchNormalization(name='bilstm2_bn'),
        Dropout(0.30, name='bilstm2_drop'),

        # ── LSTM Final — colapsa dimensión temporal ───────────────
        LSTM(64, return_sequences=False, name='lstm_final'),
        BatchNormalization(name='lstm_final_bn'),
        Dropout(0.30, name='lstm_final_drop'),

        # ── Bloque clasificador denso ─────────────────────────────
        Dense(256, activation='relu',
              kernel_regularizer=l2(1e-4), name='dense_1'),
        BatchNormalization(name='dense1_bn'),
        Dropout(0.40, name='dense1_drop'),

        Dense(128, activation='relu', name='dense_2'),

        # ── Capa de salida ────────────────────────────────────────
        Dense(num_classes, activation='softmax', name='output'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 2: BiLSTM + Atención Multi-Cabeza
# ══════════════════════════════════════════════════════════════════════════════

def build_attention_model(num_classes: int = len(ACTIONS),
                          learning_rate: float = 5e-4) -> tf.keras.Model:
    """
    BiLSTM con Mecanismo de Atención Multi-Cabeza.

    ¿Cuándo usar este modelo?
    ─────────────────────────
    El mecanismo de atención permite al modelo enfocarse en los instantes
    más relevantes dentro de la secuencia (e.g., el punto de máxima
    configuración de la mano, o el inicio del movimiento). Es especialmente
    útil para señas complejas o similares entre sí.

    Arquitectura detallada
    ──────────────────────
    Input  (batch, T, FEATURE_DIM)
      │
      ├─ TimeDistributed(Dense 256)         ← Proyección
      │   BatchNorm + Dropout(0.20)
      │
      ├─ BiLSTM(256, return_seqs=True)      ← Encoder temporal
      │   BatchNorm + Dropout(0.30)
      │
      ├─ BiLSTM(128, return_seqs=True)
      │   BatchNorm + Dropout(0.30)
      │
      ├─ MultiHeadAttention(4 cabezas, key_dim=64)
      │   └─ Conexión residual: Add([LSTM_out, Attn_out])
      │   └─ LayerNormalization
      │
      ├─ GlobalAveragePooling1D             ← Agrega sobre T
      │
      ├─ Dense(256, relu) + Dropout(0.40)
      ├─ Dense(128, relu)
      └─ Dense(num_classes, softmax)

    Args:
        num_classes:   Número de señas (clases).
        learning_rate: Tasa de aprendizaje inicial.
    Returns:
        Modelo Keras compilado.
    """
    inputs = Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name='input_seq')

    # ── Proyección temporal ──────────────────────────────────────
    x = TimeDistributed(
        Dense(256, activation='relu'), name='time_projection'
    )(inputs)
    x = BatchNormalization(name='proj_bn')(x)
    x = Dropout(0.20, name='proj_drop')(x)

    # ── Encoder BiLSTM ───────────────────────────────────────────
    x = Bidirectional(
        LSTM(256, return_sequences=True, dropout=0.20),
        name='bilstm_1'
    )(x)
    x = BatchNormalization(name='bilstm1_bn')(x)
    x = Dropout(0.30, name='bilstm1_drop')(x)

    x = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.20),
        name='bilstm_2'
    )(x)
    x = BatchNormalization(name='bilstm2_bn')(x)
    x = Dropout(0.30, name='bilstm2_drop')(x)

    # ── Atención Multi-Cabeza (auto-atención sobre secuencia) ─────
    # La atención aprende QUÉ frames son más discriminativos para cada seña.
    attn_out = MultiHeadAttention(
        num_heads=4, key_dim=64, dropout=0.10,
        name='multi_head_attention'
    )(x, x)   # query=x, value=x (self-attention)

    # Conexión residual + normalización (estilo Transformer)
    x = Add(name='residual_add')([x, attn_out])
    x = LayerNormalization(epsilon=1e-6, name='layer_norm')(x)

    # ── Agregación temporal → vector de características ──────────
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)

    # ── Clasificador ─────────────────────────────────────────────
    x = Dense(256, activation='relu',
              kernel_regularizer=l2(1e-4), name='dense_1')(x)
    x = Dropout(0.40, name='dense1_drop')(x)
    x = Dense(128, activation='relu', name='dense_2')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='LSM_Attention_LSTM')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN RÁPIDA (python model.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    sep = '═' * 65

    print(f'\n{sep}')
    print('  MODELO 1: BiLSTM con Proyección Temporal')
    print(sep)
    m1 = build_lstm_model()
    m1.summary()
    params1 = m1.count_params()

    print(f'\n{sep}')
    print('  MODELO 2: BiLSTM + Atención Multi-Cabeza')
    print(sep)
    m2 = build_attention_model()
    m2.summary()
    params2 = m2.count_params()

    print(f'\n{"─" * 65}')
    print(f'  FEATURE_DIM    : {FEATURE_DIM}')
    print(f'  SEQUENCE_LENGTH: {SEQUENCE_LENGTH}')
    print(f'  Input shape    : (batch, {SEQUENCE_LENGTH}, {FEATURE_DIM})')
    print(f'  Output shape   : (batch, {len(ACTIONS)})')
    print(f'  Clases         : {list(ACTIONS)}')
    print(f'  Parámetros M1  : {params1:,}')
    print(f'  Parámetros M2  : {params2:,}')
    print(f'{"─" * 65}\n')
