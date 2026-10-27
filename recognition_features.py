"""Preprocesamiento compartido por entrenamiento e inferencia de modelos web v2."""
import numpy as np
from config import FEATURE_DIM, POSE_DIM, FACE_KEY_DIM, FACE_FULL_DIM, USE_FULL_FACE

PREPROCESSING = 'body-relative-v1'
HAND_PREPROCESSING = 'body-hand-relative-v2'


def feature_dimension(preprocessing):
    if preprocessing == HAND_PREPROCESSING:
        return FEATURE_DIM + 128
    if preprocessing in (None, PREPROCESSING):
        return FEATURE_DIM
    raise ValueError('Preprocesamiento de modelo desconocido.')


def local_hands(points):
    """Forma relativa a la muñeca, escalada por la palma, sin reflejar ni rotar."""
    local, visible = [], []
    for hand in regions(np.asarray(points, dtype=np.float32))[-2:]:
        present = np.any(hand != 0, axis=(-2, -1))
        scale = np.linalg.norm(hand[..., 9, :] - hand[..., 0, :], axis=-1)
        present &= scale > 1e-4
        relative = (hand - hand[..., :1, :]) / np.maximum(scale[..., None, None], 1e-4)
        relative = np.where(present[..., None, None], relative, 0)
        local.append(relative.reshape(*hand.shape[:-2], 63))
        visible.append(present.astype(np.float32)[..., None])
    return np.concatenate(local + visible, axis=-1)


def prepare_sequence(points, preprocessing=HAND_PREPROCESSING):
    feature_dimension(preprocessing)
    if preprocessing is None:
        return np.asarray(points, dtype=np.float32)
    body = normalize_sequence(points)
    if preprocessing == HAND_PREPROCESSING:
        return np.concatenate([body, local_hands(points)], axis=-1)
    return body


def hand_descriptor(points):
    hands = local_hands(points)
    return np.concatenate([np.median(hands, axis=0), np.std(hands, axis=0)])


def nearest_distance(descriptor, references):
    distances = np.sqrt(np.mean((np.asarray(references) - descriptor) ** 2, axis=1))
    return float(np.mean(np.sort(distances)[:min(3, len(distances))]))


def fit_hand_guard(train_points, train_targets, val_points, val_targets):
    """Referencias solo de entrenamiento; tolerancia comprobada con validación."""
    train_desc = np.stack([hand_descriptor(p) for p in train_points])
    val_desc = np.stack([hand_descriptor(p) for p in val_points])
    classes = []
    for label in sorted(np.unique(train_targets)):
        refs = train_desc[train_targets == label]
        distances = [nearest_distance(d, np.delete(refs, i, axis=0)) for i, d in enumerate(refs)]
        val_distances = [nearest_distance(d, refs) for d in val_desc[val_targets == label]]
        threshold = max(.035, float(np.quantile(distances, .95)), max(val_distances, default=0)) * 1.25
        classes.append({'references': refs.tolist(), 'max_distance': threshold})
    return {'version': 'local-hand-shape-v1', 'classes': classes}


def hand_guard_accept(points, predicted, guard):
    if not guard:
        return True
    item = guard['classes'][int(predicted)]
    return nearest_distance(hand_descriptor(points), item['references']) <= item['max_distance']


def regions(points):
    face_dim = FACE_FULL_DIM if USE_FULL_FACE else FACE_KEY_DIM
    return [points[..., :POSE_DIM].reshape(*points.shape[:-1], 33, 4),
            points[..., POSE_DIM:POSE_DIM + face_dim].reshape(*points.shape[:-1], -1, 3),
            points[..., POSE_DIM + face_dim:POSE_DIM + face_dim + 63].reshape(*points.shape[:-1], 21, 3),
            points[..., -63:].reshape(*points.shape[:-1], 21, 3)]


def normalize_sequence(points):
    """Ancla XY al torso y escala por hombros; conserva movimiento y ausencias."""
    result = np.asarray(points, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != FEATURE_DIM or not np.isfinite(result).all():
        raise ValueError('Secuencia inválida para normalizar.')
    pose = regions(result)[0]
    shoulders = pose[:, [11, 12], :]
    widths = np.linalg.norm(shoulders[:, 0, :2] - shoulders[:, 1, :2], axis=1)
    valid = np.all(shoulders[:, :, 3] >= .4, axis=1) & (widths >= .03)
    if np.count_nonzero(valid) >= len(result) // 2:
        center = np.median(shoulders[valid, :, :2].mean(axis=1), axis=0)
        scale = float(np.median(widths[valid]))
    else:
        # Si falta el torso, mantener un marco estable en vez de inventar hombros.
        center, scale = np.array([.5, .5], dtype=np.float32), 1.0
    for region in regions(result):
        present = np.any(region[..., :3] != 0, axis=-1)
        region[..., :2] = np.where(present[..., None], (region[..., :2] - center) / scale, 0)
        region[..., 2] = np.where(present, region[..., 2] / scale, 0)
    return result


def augment_viewpoints(points, targets, seed=42):
    """Variaciones XY leves de cámara sobre datos crudos, solo para entrenar.

    Cada toma usa una transformación constante para conservar su movimiento.
    No simula giros 3D ni inventa puntos ocultos; nunca refleja las manos.
    """
    raw = np.asarray(points, dtype=np.float32)
    rng = np.random.default_rng(seed)
    output = [raw.copy()]
    for direction in (-1, 1):
        varied = raw.copy()
        for sequence in varied:
            angle = np.deg2rad(direction * rng.uniform(4, 10))
            cosine, sine = np.cos(angle), np.sin(angle)
            matrix = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
            matrix = matrix @ np.diag([rng.uniform(.95, 1.05), 1.0])
            pose = regions(sequence)[0]
            visible = np.all(pose[:, [11, 12], 3] >= .4, axis=1)
            center = (np.median(pose[visible][:, [11, 12], :2].mean(axis=1), axis=0)
                      if np.any(visible) else np.array([.5, .5]))
            for region in regions(sequence):
                present = np.any(region[..., :3] != 0, axis=-1)
                region[..., :2] = np.where(present[..., None],
                                           (region[..., :2] - center) @ matrix.T + center, 0)
        output.append(varied)
    return np.concatenate(output), np.concatenate([targets] * 3, axis=0)


def augment_training(points, targets, seed=42):
    """Originales + dos variaciones leves. No refleja, invierte ni cambia clases."""
    rng = np.random.default_rng(seed)
    output = [points.copy()]
    for _ in range(2):
        varied = points.copy()
        for sequence in varied:
            scale = rng.uniform(.95, 1.05)
            for region in regions(sequence[:, :FEATURE_DIM]):
                present = np.any(region[..., :3] != 0, axis=-1)
                noise = rng.normal(0, .005, region[..., :3].shape)
                region[..., :3] = np.where(present[..., None], region[..., :3] * scale + noise, 0)
            if sequence.shape[-1] > FEATURE_DIM:
                shapes = sequence[:, FEATURE_DIM:FEATURE_DIM + 126].reshape(len(sequence), 2, 21, 3)
                present = np.any(shapes != 0, axis=-1)
                shapes[:] = np.where(present[..., None], shapes + rng.normal(0, .005, shapes.shape), 0)
        output.append(varied)
    return np.concatenate(output), np.concatenate([targets, targets, targets], axis=0)


def decision_metrics(probabilities, targets, threshold=.85, margin=.15, allowed=None):
    probabilities = np.asarray(probabilities)
    order = np.sort(probabilities, axis=1)
    accepted = (order[:, -1] >= threshold) & ((order[:, -1] - order[:, -2]) >= margin)
    if allowed is not None:
        accepted &= np.asarray(allowed, dtype=bool)
    correct = np.argmax(probabilities, axis=1) == targets
    return {'threshold': threshold, 'margin': margin, 'evaluated': int(len(targets)),
            'accepted': int(accepted.sum()), 'coverage': float(np.mean(accepted)),
            'accepted_accuracy': float(np.mean(correct[accepted])) if accepted.any() else None}


def validation_threshold(probabilities, targets):
    # Ajustar solo con validación. El test queda reservado para el informe final.
    candidates = [decision_metrics(probabilities, targets, threshold) for threshold in [.85, .90, .95]]
    eligible = [m for m in candidates if m['accepted'] >= 3 and m['accepted_accuracy'] >= .9]
    return max(eligible, key=lambda m: m['coverage']) if eligible else candidates[-1]
