"""Entrenamiento aislado por lengua. Lo inicia la página en un subproceso."""
import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sign_languages import WEB_DATA, MIN_SAMPLES, language_info, model_dir, pipeline_spec
from web_dataset import read_dataset, split_per_class
from recognition_features import (prepare_sequence, augment_training, augment_viewpoints, validation_threshold, decision_metrics,
                                  HAND_PREPROCESSING, fit_hand_guard, hand_guard_accept)


def validation_summary(probabilities, targets, allowed, threshold):
    import numpy as np
    from sklearn.metrics import f1_score
    return {**decision_metrics(probabilities, targets, threshold, allowed=allowed),
            'accuracy': float(np.mean(np.argmax(probabilities, axis=1) == targets)),
            'macro_f1': float(f1_score(targets, np.argmax(probabilities, axis=1), average='macro', zero_division=0))}


def candidate_is_not_worse(candidate, previous):
    if previous is None:
        return True
    return all(float(candidate.get(key) or 0) + 1e-6 >= float(previous.get(key) or 0)
               for key in ('accuracy', 'macro_f1', 'accepted_accuracy', 'coverage'))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def train_language(language, epochs, job_id):
    language_info(language)
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Identificador de entrenamiento inválido.')
    progress_path = WEB_DATA / 'jobs' / f'{job_id}.json'
    status = {'id': job_id, 'language': language, 'state': 'running', 'epoch': 0,
              'epochs': epochs, 'message': 'Cargando ejemplos completos…'}
    write_json(progress_path, status)
    try:
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        from model import build_web_model

        tf.keras.utils.set_random_seed(42)
        raw, y, labels = read_dataset(language, MIN_SAMPLES)
        train, validation, test = split_per_class(y)
        X = np.stack([prepare_sequence(sequence) for sequence in raw])
        onehot = tf.keras.utils.to_categorical(y, len(labels))
        directory = model_dir(language) / job_id
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = directory / 'model.keras'
        # Estadísticas solo del entrenamiento; jamás de validación o prueba.
        mean = X[train].mean(axis=(0, 1))
        variance = np.maximum(X[train].var(axis=(0, 1)), .01)
        core = build_web_model(len(labels), input_dim=X.shape[-1])
        inputs = tf.keras.Input(shape=X.shape[1:])
        normalized = tf.keras.layers.Normalization(mean=mean, variance=variance)(inputs)
        model = tf.keras.Model(inputs, core(normalized), name='Sign_Web_Normalized')
        # El optimizador legacy de TF 2.15 evita la ruta lenta de Adam en Apple Silicon.
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5e-4, clipnorm=1.0),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        # Separar primero: ninguna variante puede filtrarse a validación/prueba.
        view_points, view_targets = augment_viewpoints(raw[train], onehot[train])
        view_features = np.stack([prepare_sequence(sequence) for sequence in view_points])
        X_fit, y_fit = augment_training(view_features, view_targets)

        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                status.update(epoch=epoch + 1, message='Entrenando y evaluando la validación.',
                              metrics={k: float(v) for k, v in (logs or {}).items()})
                write_json(progress_path, status)

        counts = np.bincount(y[train], minlength=len(labels))
        weights = {i: float(len(train) / (len(labels) * n)) for i, n in enumerate(counts)}
        model.fit(X_fit, y_fit, validation_data=(X[validation], onehot[validation]),
                  epochs=epochs, batch_size=min(16, len(train)), verbose=2, class_weight=weights,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=.5),
                      Progress(),
                  ])
        model = tf.keras.models.load_model(checkpoint, compile=False, safe_mode=True)
        val_probs = model(X[validation], training=False).numpy()
        calibration = validation_threshold(val_probs, y[validation])
        guard = fit_hand_guard(view_points, np.argmax(view_targets, axis=1), raw[validation], y[validation])
        val_allowed = [hand_guard_accept(p, i, guard) for p, i in zip(raw[validation], np.argmax(val_probs, axis=1))]
        candidate_validation = validation_summary(val_probs, y[validation], val_allowed, calibration['threshold'])
        previous_validation, previous_version = None, None
        comparison_note = 'Primer modelo para este vocabulario.'
        manifest = model_dir(language) / 'current.json'
        if manifest.exists():
            previous_version = json.loads(manifest.read_text())['version']
            if len(previous_version) != 32 or any(c not in '0123456789abcdef' for c in previous_version):
                raise ValueError('Versión previa de modelo inválida; se conserva sin reemplazar.')
            previous_path = model_dir(language) / previous_version
            old_meta = json.loads((previous_path / 'metadata.json').read_text())
            if ([label['id'] for label in old_meta['labels']] == [label['id'] for label in labels]
                    and old_meta['pipeline'] == pipeline_spec()):
                old_model = tf.keras.models.load_model(previous_path / 'model.keras', compile=False, safe_mode=True)
                old_input = np.stack([prepare_sequence(p, old_meta.get('preprocessing')) for p in raw[validation]])
                old_probs = old_model(old_input, training=False).numpy()
                old_allowed = [hand_guard_accept(p, i, old_meta.get('hand_guard')) for p, i in zip(raw[validation], np.argmax(old_probs, axis=1))]
                previous_validation = validation_summary(old_probs, y[validation], old_allowed, old_meta.get('decision_threshold', .85))
                comparison_note = 'Ambas versiones evaluadas en las mismas secuencias de validación; prueba excluida de la selección.'
            else:
                comparison_note = 'Vocabulario o configuración distintos: no se comparan porcentajes entre versiones.'
        activated = candidate_is_not_worse(candidate_validation, previous_validation)
        probabilities = model(X[test], training=False).numpy()
        pred = np.argmax(probabilities, axis=1)
        test_allowed = [hand_guard_accept(p, i, guard) for p, i in zip(raw[test], pred)]
        report = classification_report(y[test], pred, labels=list(range(len(labels))),
                                       target_names=[x['id'] for x in labels],
                                       output_dict=True, zero_division=0)
        metadata = {
            'language': language, 'pipeline': pipeline_spec(), 'labels': labels,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'samples': len(y), 'split': {'train': len(train), 'validation': len(validation), 'test': len(test)},
            'test_accuracy': float(np.mean(pred == y[test])), 'report': report,
            'preprocessing': HAND_PREPROCESSING, 'architecture': 'compact-bilstm-hand-v3',
            'input_dim': int(X.shape[-1]), 'hand_guard': guard,
            'dataset_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
            'augmentation': {'training_only': True, 'variants_per_original': 8,
                             'camera_roll_degrees': [4, 10], 'horizontal_scale': [.95, 1.05]},
            'validation_decisions': candidate_validation,
            'test_decisions': decision_metrics(probabilities, y[test], calibration['threshold'], allowed=test_allowed),
            'model_comparison': {'previous_version': previous_version, 'previous_validation': previous_validation,
                                 'candidate_validation': candidate_validation, 'note': comparison_note},
            'activated': activated,
            'decision_threshold': calibration['threshold'], 'decision_margin': .15,
            'confusion_matrix': confusion_matrix(y[test], pred, labels=list(range(len(labels)))).tolist(),
            'evaluation_note': 'Prueba con secuencias reservadas del mismo dataset; no demuestra rendimiento con nuevos usuarios.',
        }
        write_json(directory / 'metadata.json', metadata)
        # Solo activar después de guardar modelo y metadatos completos.
        if activated:
            write_json(manifest, {'version': job_id})
        status.update(state='completed', activated=activated,
                      message='Modelo entrenado, comparado y activado. Ya puedes probarlo.' if activated else
                      'Se conserva el modelo anterior: la nueva versión empeoró al menos una métrica de validación.', result=metadata)
        write_json(progress_path, status)
    except Exception as exc:
        status.update(state='failed', message=str(exc))
        write_json(progress_path, status)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    if not 5 <= args.epochs <= 200:
        parser.error('Las épocas deben estar entre 5 y 200.')
    train_language(args.language, args.epochs, args.job_id)
