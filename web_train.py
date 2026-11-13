"""Entrenamiento aislado por lengua. Lo inicia la página en un subproceso."""
import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sign_languages import WEB_DATA, MIN_SAMPLES, language_info, model_dir, pipeline_spec
from studio_services import load_inference_model
from web_dataset import read_dataset, split_per_class
from recognition_features import (prepare_sequence, augment_training, augment_viewpoints, validation_threshold, decision_metrics,
                                  HAND_PREPROCESSING, STABLE_PREPROCESSING, FINGER_PREPROCESSING, fit_hand_guard, hand_guard_accept,
                                  fit_temporal_guard, temporal_guard_accept, calibrate_decision)
from dataset_tools import TRAIN_SUPPORT_TARGET, training_provenance, grouped_split, persistent_split
from posture_vote import apply_vote, fit_vote
from motion_features import fit_motion_profile, augment_timing, motion_challenges
from validation_protocol import calibration_split, validation_support
from recognition_decision import batch_evidence, calibrate_joint_policy, policy_allowed, rejection_report, decision_details
from recognition_features import CHEST_PREPROCESSING, DISCRIMINATIVE_PREPROCESSING, augment_articulation, augment_robust
from recognition_analysis import (evaluate, temperature_probabilities, fit_temperature, calibrate_policy,
    dataset_summary, deduplicate, regression_check, preprocessing_probe, label_review_candidates)


# Incluye clases anteriores que conservan ejemplos para no entrenar solamente el vocabulario nuevo.
def retained_label_ids(language, label_ids):
    """Rehearsal uses all retained classes from the active model, never just new ones."""
    if label_ids is None: return None
    manifest=model_dir(language)/'current.json'
    if not manifest.exists():return label_ids
    selection=json.loads(manifest.read_text(encoding='utf-8'))
    from web_dataset import validate_identifier, inventory
    version=selection.get('version');validate_identifier(version)
    meta=json.loads((model_dir(language)/version/'metadata.json').read_text(encoding='utf-8'))
    if meta['pipeline'] != pipeline_spec():return label_ids
    counts={r['id']:r['samples'] for r in inventory(language)}
    retained=[l['id'] for l in meta['labels'] if counts.get(l['id'],0)>0]
    return list(dict.fromkeys(list(label_ids)+retained))



# Resume aciertos, F1 y aceptación para comparar resultados de validación.
def validation_summary(probabilities, targets, allowed, threshold, margin=.15):
    import numpy as np
    from sklearn.metrics import f1_score
    return {**decision_metrics(probabilities, targets, threshold, margin, allowed=allowed),
            'accuracy': float(np.mean(np.argmax(probabilities, axis=1) == targets)),
            'macro_f1': float(f1_score(targets, np.argmax(probabilities, axis=1), average='macro', zero_division=0))}


# Calcula precisión de aceptaciones, recuperación y cobertura separadas por clase.
def class_decisions(probabilities, targets, allowed, threshold, margin, labels):
    """Accepted precision is by predicted class; recall/coverage by true class."""
    import numpy as np
    predicted = np.argmax(probabilities, axis=1)
    order = np.sort(probabilities, axis=1)
    accepted = (np.asarray(allowed, bool) & (order[:, -1] >= threshold)
                & ((order[:, -1] - order[:, -2]) >= margin))
    output = []
    for i, label in enumerate(labels):
        true = targets == i
        chosen = accepted & (predicted == i)
        output.append({'id': label['id'], 'text': label['text'],
                       'support': int(true.sum()), 'accepted': int(chosen.sum()),
                       'recall': float(np.mean(predicted[true] == i)) if true.any() else 0.,
                       'coverage': float(np.mean(accepted[true])) if true.any() else 0.,
                       'precision': float(np.mean(targets[chosen] == i)) if chosen.any() else None})
    return output


# Compara métricas globales del candidato y el anterior cuando existe una referencia comparable.
def candidate_is_not_worse(candidate, previous):
    if previous is None:
        return True
    return all(float(candidate.get(key) or 0) + 1e-6 >= float(previous.get(key) or 0)
               for key in ('accuracy', 'macro_f1', 'accepted_accuracy', 'coverage'))


# Escribe progreso o metadatos en un archivo temporal y lo sustituye atómicamente.
def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


# Ejecuta separación, aumentos, entrenamiento, calibración, evaluación y posible activación de una versión.
def train_language(language, epochs, job_id, label_ids=None, strategy="baseline", activate=True):
    language_info(language)
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Identificador de entrenamiento inválido.')
    progress_path = WEB_DATA / 'jobs' / f'{job_id}.json'
    status = {'id': job_id, 'language': language, 'state': 'running', 'epoch': 0,
              'epochs': epochs, 'pid': os.getpid(), 'message': 'Cargando ejemplos completos…'}
    write_json(progress_path, status)
    try:
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
        from model import build_web_model

        tf.keras.utils.set_random_seed(42)
        if strategy not in ('baseline', 'discriminative', 'contrastive', 'trajectory', 'robust', 'torso'):
            raise ValueError('Estrategia de entrenamiento desconocida.')
        label_ids = retained_label_ids(language, label_ids)
        raw, y, labels = read_dataset(language, MIN_SAMPLES, label_ids)
        raw, y, removed_duplicates = deduplicate(raw, y)
        if np.min(np.bincount(y)) < MIN_SAMPLES:
            raise ValueError('Faltan ejemplos únicos por clase después de quitar duplicados.')
        groups, provenance = training_provenance(language, raw, y, labels)
        train, validation, test = persistent_split(y, groups, model_dir(language)/'evaluation_split.json', fingerprints=[hashlib.sha256(p.tobytes()).hexdigest() for p in raw], label_names=[label['text'] for label in labels])
        counts = np.bincount(y[train], minlength=len(labels))
        low_support = [f'{label["text"]} ({count})'
                       for label, count in zip(labels, counts) if count < TRAIN_SUPPORT_TARGET]
        if low_support:
            status['data_warning'] = (
                'Estas señas tienen pocas tomas únicas para aprender. ' + ', '.join(low_support)
                + f'. Busca al menos {TRAIN_SUPPORT_TARGET} por seña en entrenamiento.'
                ' Graba más ejemplos en nuevas sesiones. Las tomas reservadas para evaluar se conservan.'
            )
            write_json(progress_path, status)
        selection, calibration_indices, protocol = calibration_split(y, groups, validation,
            model_dir(language)/'calibration_split.json',
            fingerprints=[hashlib.sha256(p.tobytes()).hexdigest() for p in raw])
        protocol.update(independent_calibration=protocol['independent'],
                        selection_samples=len(selection), calibration_samples=len(calibration_indices))
        dataset_diagnostics = dataset_summary(raw, y, labels, groups, provenance)
        preprocessing = {
            'baseline': FINGER_PREPROCESSING,
            'torso': CHEST_PREPROCESSING,
        }.get(strategy, DISCRIMINATIVE_PREPROCESSING)
        X = np.stack([prepare_sequence(sequence, preprocessing) for sequence in raw])
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
        # Derive every feature from the SAME augmented landmarks. Never perturb
        # normalized coordinates independently of finger geometry.
        fit_points, y_fit = (augment_robust if strategy == 'robust' else augment_timing if strategy in ('baseline', 'trajectory', 'torso') else augment_articulation)(view_points, view_targets)
        X_fit = np.stack([prepare_sequence(sequence, preprocessing) for sequence in fit_points])

        from discriminative_learning import HardExampleSequence, MineHardExamples, contrastive_warmup
        contrastive_history = []
        if strategy == 'contrastive':
            status['message'] = 'Separando representaciones con aprendizaje contrastivo (solo entrenamiento).'
            write_json(progress_path, status)
            embedding_core = tf.keras.Model(core.input, core.layers[-3].output)
            encoder = tf.keras.Model(inputs, embedding_core(normalized))
            contrastive_history = contrastive_warmup(encoder, X[train], y[train], epochs=min(4, max(1, epochs//4)))
        history = []

        # Calcula una métrica de selección que considera F1 macro y la clase con menor recuperación.
        class BalancedValidation(tf.keras.callbacks.Callback):
            # Evalúa la reserva de selección y añade las métricas equilibradas a los registros de la época.
            def on_epoch_end(self, epoch, logs=None):
                probabilities = self.model(X[selection], training=False).numpy()
                predicted = np.argmax(probabilities, axis=1)
                macro = float(f1_score(y[selection], predicted, average='macro', zero_division=0))
                worst = float(np.min(recall_score(y[selection], predicted,
                                                  labels=np.arange(len(labels)), average=None, zero_division=0)))
                logs['val_macro_f1'] = macro
                logs['val_min_recall'] = worst
                # Tiny loss term breaks metric ties; worst class cannot hide
                # behind high accuracy in the easy or overrepresented classes.
                logs['val_balanced_score'] = .7 * macro + .3 * worst - 1e-4 * min(float(logs['val_loss']), 100)

        # Publica en disco el avance que la interfaz consulta durante el entrenamiento.
        class Progress(tf.keras.callbacks.Callback):
            # Añade métricas e historia de la época y actualiza el trabajo de entrenamiento.
            def on_epoch_end(self, epoch, logs=None):
                status.update(epoch=epoch + 1, message='Entrenando y evaluando la validación.',
                              metrics={k: float(v) for k, v in (logs or {}).items()})
                history.append({'epoch': epoch + 1, **status['metrics']})
                status['history'] = history
                write_json(progress_path, status)

        weights = {i: float(len(train) / (len(labels) * n)) for i, n in enumerate(counts)}
        callbacks = [BalancedValidation(),
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_balanced_score', mode='max', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_balanced_score', mode='max', min_delta=1e-6, patience=16, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=.5)]
        initial_weights = model.get_weights()
        if strategy in ('baseline', 'trajectory', 'robust', 'torso'):
            fit_args = dict(x=X_fit, y=y_fit, batch_size=min(16,len(train)), class_weight=weights)
            fit_callbacks = callbacks+[Progress()]
        else:
            fit_data=HardExampleSequence(X_fit,y_fit,min(16,len(train)))
            fit_args = dict(x=fit_data, workers=1, use_multiprocessing=False)
            fit_callbacks = callbacks+[MineHardExamples(fit_data),Progress()]
        common = dict(validation_data=(X[selection],onehot[selection]),epochs=epochs,
                      verbose=2,callbacks=fit_callbacks)
        recovery = None
        try:
            model.fit(**fit_args, **common)
        except tf.errors.InvalidArgumentError as error:
            if 'var and grad do not have the same shape' not in str(error):
                raise
            # Un paso fallido puede haber actualizado solo algunas variables.
            # Recuperar pesos y slots de un checkpoint completo antes de seguir.
            if checkpoint.exists():
                model = tf.keras.models.load_model(checkpoint, safe_mode=True)
                start_epoch = int(max(history,key=lambda row: row['val_balanced_score'])['epoch'])
                optimizer = model.optimizer
                history[:] = [row for row in history if row['epoch'] <= start_epoch]
            else:
                model.set_weights(initial_weights)
                start_epoch = 0
                history.clear()
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4,clipnorm=1.0)
            recovery = dict(reason='invalid_recurrent_gradient_shape',
                            resumed_epoch=start_epoch,execution='eager')
            status.update(message='Recuperando el mejor checkpoint. El entrenamiento continúa en modo compatible.',
                          recovery=recovery,history=history,epoch=start_epoch)
            write_json(progress_path,status)
            model.compile(optimizer=optimizer,loss='categorical_crossentropy',
                          metrics=['accuracy'],run_eagerly=True)
            model.fit(**fit_args, **common, initial_epoch=start_epoch)
        model = load_inference_model(checkpoint)
        val_probs = model(X[validation], training=False).numpy()
        cal_probs = model(X[calibration_indices], training=False).numpy()
        guard = fit_hand_guard(view_points, np.argmax(view_targets, axis=1), raw[calibration_indices], y[calibration_indices], contrastive=True, descriptor_version='finger-articulation-v3')
        temporal_guard = fit_temporal_guard(raw[train], y[train], raw[calibration_indices], y[calibration_indices], descriptor_version='trajectory-v2')
        motion_profile = fit_motion_profile(raw[train], y[train])
        temperature = fit_temperature(cal_probs, y[calibration_indices])
        cal_probs = temperature_probabilities(cal_probs, temperature)
        val_probs = temperature_probabilities(val_probs, temperature)
        select_probs = temperature_probabilities(model(X[selection], training=False).numpy(), temperature)
        vote = fit_vote(X[train], y[train], X[selection], y[selection], select_probs)
        cal_probs = apply_vote(X[calibration_indices], cal_probs, vote)
        val_probs = apply_vote(X[validation], val_probs, vote)
        guard_metadata = {'hand_guard': guard, 'temporal_guard': temporal_guard}
        cal_scores = batch_evidence(raw[calibration_indices], cal_probs, guard_metadata)
        policy = calibrate_joint_policy(cal_probs, y[calibration_indices], cal_scores, guard, groups[calibration_indices], motion_guard=temporal_guard)
        val_scores = batch_evidence(raw[validation], val_probs, guard_metadata)
        val_allowed = policy_allowed(val_probs, policy, val_scores)
        calibration = {'threshold': policy['threshold'], 'margin': policy['margin']}
        validation_evaluation = evaluate(val_probs, y[validation], labels, policy, val_allowed)
        candidate_validation = validation_summary(val_probs, y[validation], val_allowed, calibration['threshold'], calibration['margin'])
        previous_validation, previous_version = None, None
        comparison_note = 'Primer modelo para este vocabulario.'
        previous_bundle = None
        manifest = model_dir(language) / 'current.json'
        if manifest.exists():
            previous_version = json.loads(manifest.read_text(encoding='utf-8')).get('version')
            from web_dataset import validate_identifier
            if previous_version:
                validate_identifier(previous_version)
                previous_path = model_dir(language) / previous_version
                old_meta = json.loads((previous_path / 'metadata.json').read_text(encoding='utf-8'))
                if old_meta['pipeline'] == pipeline_spec():
                    old_model = load_inference_model(previous_path / 'model.keras')
                    previous_bundle = (old_model, old_meta)
                comparison_note = 'Se comparan clases comunes solo en ejemplos ajenos al entrenamiento y calibración anteriores.'
        class_validation = [{**row, 'precision': row['accepted_precision']} for row in validation_evaluation['by_class']]
        probabilities = temperature_probabilities(model(X[test], training=False).numpy(), temperature)
        probabilities = apply_vote(X[test], probabilities, vote)
        pred = np.argmax(probabilities, axis=1)
        test_scores = batch_evidence(raw[test], probabilities, guard_metadata)
        test_allowed = policy_allowed(probabilities, policy, test_scores)
        report = classification_report(y[test], pred, labels=list(range(len(labels))),
                                       target_names=[x['id'] for x in labels],
                                       output_dict=True, zero_division=0)
        test_evaluation = evaluate(probabilities, y[test], labels, policy, test_allowed)
        train_probs = temperature_probabilities(model(X[train], training=False).numpy(), temperature)
        train_probs = apply_vote(X[train], train_probs, vote)
        train_evaluation = evaluate(train_probs, y[train], labels, policy)
        comparison = {'previous_version':previous_version,'note':comparison_note,'comparable':False,'passed':False}
        test_comparison = {'comparable':False,'note':'No hay una prueba independiente común documentada.'}
        if previous_bundle:
            old_model,old_meta=previous_bundle
            old_ids=[l['id'] for l in old_meta['labels']]
            # Compara modelos solo en ejemplos comunes ajenos a sus entrenamientos y calibraciones.
            def compare_previous(indices_set, candidate_probs, candidate_allowed, final_test=False):
                excluded_splits=('train','validation') if final_test else ('train',)
                excluded=set(h for part in excluded_splits for h in old_meta.get('split_hashes',{}).get(part,[]))
                old_groups=set(g for part in excluded_splits for g in old_meta.get('split_groups',{}).get(part,[]))
                common=[j for j,k in enumerate(indices_set) if labels[y[k]]['id'] in old_ids
                        and hashlib.sha256(raw[k].tobytes()).hexdigest() not in excluded and str(groups[k]) not in old_groups]
                if not excluded or not old_meta.get('split_groups') or not common:
                    return {'comparable':False,'passed':False,'note':'Falta procedencia o reserva común independiente suficiente.'}
                indices=indices_set[common];old_y=np.array([old_ids.index(labels[y[k]]['id']) for k in indices])
                old_x=np.stack([prepare_sequence(raw[k],old_meta.get('preprocessing')) for k in indices])
                old_p=temperature_probabilities(old_model(old_x,training=False).numpy(),old_meta.get('temperature',1.))
                old_p=apply_vote(old_x,old_p,old_meta.get('posture_vote'))
                old_allowed=[decision_details(raw[k],p,old_meta)['accepted'] for k,p in zip(indices,old_p)]
                previous_result=evaluate(old_p,old_y,old_meta['labels'],old_meta.get('decision_policy',{'threshold':old_meta.get('decision_threshold',.85),'margin':old_meta.get('decision_margin',.15)}),old_allowed)
                candidate_result=evaluate(candidate_probs[common],y[indices],labels,policy,np.asarray(candidate_allowed)[common])
                common_rows=[row for row in candidate_result['by_class'] if row['id'] in old_ids]
                candidate_result['macro_f1']=float(np.mean([row['f1-score'] for row in common_rows]))
                result=regression_check(candidate_result,previous_result)
                result['passed'] &= all(row['support']>=3 for row in previous_result['by_class'])
                return {**result,'candidate':candidate_result,'previous':previous_result,'samples':len(common),
                        'split':'test' if final_test else 'validation',
                        'note':'Prueba excluida del ajuste y selección.' if final_test else 'Validación ajena al ajuste de pesos; se reutiliza para calibración y selección.'}
            comparison.update(compare_previous(validation,val_probs,val_allowed))
            test_comparison=compare_previous(test,probabilities,test_allowed,True)
        diagnostics = {
            'dataset':dataset_diagnostics, 'duplicates_removed':removed_duplicates,
            'preprocessing_probe':preprocessing_probe(raw,y,train,validation),
            'label_review':label_review_candidates(raw,y,labels,train),
            'train_macro_f1':train_evaluation['macro_f1'],
            'generalization_gap':train_evaluation['macro_f1']-validation_evaluation['macro_f1'],
            'causes':[{'area':'datos','evidence':provenance['note']},
                      {'area':'preprocesamiento','evidence':'Se conserva lateralidad y orientación; nuevas articulaciones y velocidades con versión independiente.'},
                      {'area':'arquitectura','evidence':'La diferencia entrenamiento/validación y las alternativas ayudan a distinguir capacidad de sobreajuste; no prueban causalidad.'},
                      {'area':'inferencia','evidence':'Rechazos y errores aceptados se evalúan por separado; falta un conjunto real de señas desconocidas.'}],
            'needs_more_data': [r['id'] for r in validation_evaluation['by_class'] if r['recall']<.8 or r['coverage']<.5 or r['support']<5],
        }
        # El estudio usa la versión más reciente; las métricas se conservan para revisión.
        activated = bool(activate)
        metadata = {
            'language': language, 'pipeline': pipeline_spec(), 'labels': labels,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'split_hashes': {name: [hashlib.sha256(p.tobytes()).hexdigest() for p in raw[indices]] for name, indices in (('train',train),('validation',validation),('test',test))},
            'samples': len(y), 'split': {'train': len(train), 'validation': len(validation), 'test': len(test)},
            'test_accuracy': float(np.mean(pred == y[test])), 'report': report,
            'preprocessing': preprocessing, 'architecture': 'compact-bilstm-articulation-v6',
            'training_recovery':recovery, 'strategy': strategy, 'posture_vote':vote, 'temperature':temperature, 'decision_policy':policy,
            'decision_input': 'single-window-v1', 'validation_protocol': protocol,
            'recognition_diagnostics': {
                'validation': rejection_report(val_probs, y[validation], labels, policy, val_scores),
                'calibration': rejection_report(cal_probs, y[calibration_indices], labels, policy, cal_scores),
                'test': rejection_report(probabilities, y[test], labels, policy, test_scores)},
            'evaluation_support': validation_support(y[test], groups[test], labels, pred,
                __import__('recognition_analysis').accepted_mask(probabilities, policy, test_allowed)),
            'validation_evaluation':validation_evaluation, 'test_evaluation':test_evaluation,
            'diagnostics':diagnostics, 'contrastive_history':contrastive_history,
            'split_groups':{name:sorted(set(groups[indices].tolist())) for name,indices in (('train',train),('validation',validation),('test',test))},
            'provenance': provenance, 'history': history, 'temporal_guard': temporal_guard, 'motion_profile': motion_profile,
            'motion_evaluation': motion_challenges(raw[test], y[test], motion_profile, temporal_guard, labels) if strategy in ('baseline', 'trajectory', 'robust', 'torso') else None,
            'input_dim': int(X.shape[-1]), 'hand_guard': guard,
            'dataset_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
            'augmentation': {'training_only': True, 'variants_per_original': 8,
                             'jitter': 'distance-timing-articulation-v1' if strategy == 'robust' else 'ordered-time-warp' if strategy in ('baseline', 'trajectory', 'torso') else 'palm-scaled-temporally-correlated',
                             'temporal_warp': {'exponents': [.8, 1.2], 'end_hold_frames': 4, 'direction_preserved': True} if strategy in ('baseline', 'trajectory', 'robust', 'torso') else None,
                             'distance_scale': [.72, 1.12] if strategy == 'robust' else None,
                             'camera_roll_degrees': [4, 10], 'horizontal_scale': [.95, 1.05]},
            'validation_decisions': validation_evaluation,
            'validation_by_class': class_validation,
            'checkpoint_selection': '0.7 * validation macro F1 + 0.3 * minimum class recall; loss breaks ties',
            'test_decisions': test_evaluation,
            'model_comparison': comparison, 'independent_test_comparison':test_comparison,
            'activated': activated, 'auto_activate': bool(activate),
            'decision_threshold': calibration['threshold'], 'decision_margin': calibration['margin'], 'stability_frames': 5,
            'confusion_matrix': confusion_matrix(y[test], pred, labels=list(range(len(labels)))).tolist(),
            'evaluation_note': 'Prueba con secuencias reservadas del mismo dataset; no demuestra rendimiento con nuevos usuarios.',
        }
        write_json(directory / 'metadata.json', metadata)
        # Solo activar después de guardar modelo y metadatos completos.
        if activated:
            write_json(manifest, {'version': job_id})
        status.update(state='completed', activated=activated,
                      message='Modelo entrenado, comparado y activado. Ya puedes probarlo.' if activated else
                      'Versión guardada sin activar por solicitud del entrenamiento.', result=metadata)
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
    parser.add_argument('--labels', nargs='+', help='IDs de las señas seleccionadas; cada una requiere el mínimo real.')
    parser.add_argument('--strategy', choices=['baseline','discriminative','contrastive','trajectory','robust','torso'], default='baseline')
    parser.add_argument('--no-activate', action='store_true', help='Guardar evaluación sin cambiar el modelo activo.')
    args = parser.parse_args()
    if not 5 <= args.epochs <= 200:
        parser.error('Las épocas deben estar entre 5 y 200.')
    train_language(args.language, args.epochs, args.job_id, args.labels, args.strategy, not args.no_activate)
