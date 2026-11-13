"""Grouped selection/calibration reserves and honest evaluation support.

The outer train/validation/test partition remains owned by dataset_tools.
This module divides ONLY its validation indices. Small datasets explicitly reuse
validation, without claiming independent calibration or changing a frozen ledger.
"""
import itertools
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


# Separa selección y calibración dentro de validación; declara explícitamente cuando deben compartir datos.
def calibration_split(targets, groups, validation, path, fingerprints=None):
    """Return original indices ``(selection, calibration, info)``.

    Each role needs at least three sequences and two independent groups per
    class. Whole groups and identical fingerprints keep their frozen role when
    the vocabulary grows. If this is impossible, both outputs are the original
    validation set and ``info['mode'] == 'shared_validation'``. No ledger is
    written on fallback. Group independence does not imply person independence.
    """
    y = np.asarray(targets)
    g = np.asarray(groups).astype(str)
    indices = np.asarray(validation, dtype=int)
    if y.ndim != 1 or g.shape != y.shape or indices.ndim != 1:
        raise ValueError('Dimensiones de validación incompatibles.')
    if not len(indices) or len(set(indices.tolist())) != len(indices) or np.any(indices < 0) or np.any(indices >= len(y)):
        raise ValueError('Índices de validación vacíos, repetidos o inválidos.')
    selected_groups = set(g[indices])
    outside = np.ones(len(y), bool)
    outside[indices] = False
    if selected_groups & set(g[outside]):
        raise ValueError('Un grupo de validación aparece también en entrenamiento o prueba.')
    hashes = None if fingerprints is None else np.asarray(fingerprints).astype(str)
    if hashes is not None:
        if hashes.shape != y.shape:
            raise ValueError('Huellas de validación incompatibles.')
        if set(hashes[indices]) & set(hashes[outside]):
            raise ValueError('Una secuencia de validación aparece también en entrenamiento o prueba.')
    classes = np.unique(y)
    path = Path(path)
    stored = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'version': 1, 'groups': {}, 'samples': {}}
    assignments = dict(stored.get('groups', {}))
    sample_roles = dict(stored.get('samples', {}))
    if stored.get('version') != 1 or any(role not in ('selection', 'calibration') for role in [*assignments.values(), *sample_roles.values()]):
        raise ValueError('Registro de selección/calibración inválido.')

    # Cuenta ejemplos y grupos por clase para describir la independencia de la reserva.
    def describe(part):
        return {'samples': len(part), 'groups': len(set(g[part])),
                'by_class': [{'class_index': int(c), 'samples': int(np.sum(y[part] == c)),
                              'groups': len(set(g[part][y[part] == c]))} for c in classes]}

    # Devuelve validación compartida con su motivo cuando no hay datos para dos reservas distintas.
    def fallback(reason):
        return indices.copy(), indices.copy(), {
            'mode': 'shared_validation', 'independent': False, 'reason': reason,
            'minimum_samples_per_class_per_role': 3, 'minimum_groups_per_class_per_role': 2,
            'selection': describe(indices), 'calibration': describe(indices),
            'note': 'Datos insuficientes para reservas separadas: selección y calibración comparten validación. La prueba permanece aparte.'}

    # A copied recording never supplies an additional independent group.
    parent = {group: group for group in selected_groups}
    # Agrupa copias de una secuencia para impedir que entren en roles distintos.
    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    if hashes is not None:
        first = {}
        for i in indices:
            if hashes[i] in first:
                parent[root(g[i])] = root(first[hashes[i]])
            else:
                first[hashes[i]] = g[i]
    units = np.array([root(group) for group in g[indices]])
    unique = sorted(set(units))
    locked = {}
    for unit in unique:
        members = indices[units == unit]
        roles = {assignments[group] for group in set(g[members]) if group in assignments}
        if hashes is not None:
            roles.update(sample_roles[h] for h in hashes[members] if h in sample_roles)
        if len(roles) > 1:
            return fallback('historical_role_conflict')
        if roles:
            locked[unit] = roles.pop()
    for c in classes:
        mask = y[indices] == c
        if np.sum(mask) < 6 or len(set(units[mask])) < 4:
            return fallback('insufficient_groups_or_samples')

    free = [group for group in unique if group not in locked]
    rng = np.random.default_rng(42)
    choices = itertools.product((0, 1), repeat=len(free)) if len(free) <= 12 else (rng.integers(0, 2, len(free)) for _ in range(2400))
    best = None
    best_score = float('inf')
    for bits in choices:
        trial = {**locked, **{group: ('selection', 'calibration')[int(bit)] for group, bit in zip(free, bits)}}
        calibration_mask = np.array([trial[unit] == 'calibration' for unit in units])
        parts = (indices[~calibration_mask], indices[calibration_mask])
        counts = np.array([[np.sum(y[part] == c) for c in classes] for part in parts])
        group_counts = np.array([[len(set(units[mask & (y[indices] == c)])) for c in classes]
                                 for mask in (~calibration_mask, calibration_mask)])
        if np.any(counts < 3) or np.any(group_counts < 2):
            continue
        score = float(np.abs(counts[0] / counts.sum(axis=0) - .5).mean()
                      + .25 * np.abs(group_counts[0] / group_counts.sum(axis=0) - .5).mean())
        if score < best_score:
            best, best_score = (trial, parts), score
    if best is None:
        return fallback('frozen_reserve_needs_more_groups')
    trial, parts = best
    assignments.update({group: trial[root(group)] for group in selected_groups})
    if hashes is not None:
        sample_roles.update({hashes[i]: assignments[g[i]] for i in indices})
    payload = {'version': 1, 'groups': assignments, 'samples': sample_roles}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name, suffix='.tmp', delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        temporary = stream.name
    try:
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return *parts, {
        'mode': 'grouped_selection_calibration', 'independent': True,
        'minimum_samples_per_class_per_role': 3, 'minimum_groups_per_class_per_role': 2,
        'selection': describe(parts[0]), 'calibration': describe(parts[1]),
        'note': 'Selección y calibración usan grupos distintos; la independencia entre personas depende de la procedencia del dataset.'}


# Calcula un intervalo orientativo de incertidumbre para un conteo de éxitos sobre observaciones.
def wilson_interval(successes, total, z=1.959963984540054):
    """95% Wilson interval for a binary event; no observation gives no estimate."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError('Conteos inválidos para el intervalo de Wilson.')
    if total == 0:
        return {'estimate': None, 'lower': None, 'upper': None, 'successes': 0, 'total': 0}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {'estimate': p, 'lower': max(0., center - radius), 'upper': min(1., center + radius),
            'successes': int(successes), 'total': int(total)}


# Describe soporte e incertidumbre por muestras y grupos sin afirmar independencia entre personas.
def validation_support(targets, groups, labels, predicted=None, accepted=None):
    """Report counts and uncertainty, by true class for coverage and predicted
    class for accepted precision. Intervals over recordings are nominal IID
    intervals, explicitly flagged when sessions repeat. Group-level intervals
    describe binary GROUP events, not a confidence interval for frame accuracy.
    Callers must also disclose legacy/unknown provenance separately.
    """
    y = np.asarray(targets)
    g = np.asarray(groups).astype(str)
    if y.ndim != 1 or g.shape != y.shape:
        raise ValueError('Dimensiones de soporte incompatibles.')
    if predicted is None:
        pred = None
        mask = None
    else:
        pred = np.asarray(predicted)
        mask = np.ones(len(y), bool) if accepted is None else np.asarray(accepted, bool)
        if pred.shape != y.shape or mask.shape != y.shape:
            raise ValueError('Dimensiones de decisiones incompatibles.')

    # Calcula soporte, cobertura e intervalos de una clase o del total evaluado.
    def row(truth, chosen):
        true_groups = set(g[truth])
        result = {'samples': int(truth.sum()), 'groups': len(true_groups),
                  'repeated_groups': len(true_groups) < int(truth.sum())}
        if pred is None:
            return result
        correct = pred == y
        result.update(accepted=int(chosen.sum()), accepted_groups=len(set(g[chosen])),
                      accepted_precision=wilson_interval(int((chosen & correct).sum()), int(chosen.sum())),
                      coverage=wilson_interval(int((truth & mask).sum()), int(truth.sum())),
                      accepted_recall=wilson_interval(int((truth & mask & correct).sum()), int(truth.sum())))
        result['sequence_intervals_assume_independence'] = True
        result['sequence_intervals_independent'] = len(set(g[truth | chosen])) == int((truth | chosen).sum())
        accepted_groups = set(g[chosen])
        safe_groups = sum(bool(np.all(correct[chosen & (g == group)])) for group in accepted_groups)
        covered_groups = sum(bool(np.any(truth & mask & (g == group))) for group in true_groups)
        result['group_intervals'] = {
            'groups_without_false_accepts': wilson_interval(safe_groups, len(accepted_groups)),
            'groups_with_any_acceptance': wilson_interval(covered_groups, len(true_groups)),
            'unit': 'recording_group',
            'note': 'Eventos por grupo, no precisión por fotograma. Si los grupos comparten persona, todavía pueden estar correlacionados.'}
        return result
    output = {'overall': row(np.ones(len(y), bool), mask), 'by_class': [],
              'confidence_level': .95,
              'note': 'Intervalos orientativos: pocas sesiones producen incertidumbre amplia. Los fotogramas no se cuentan como observaciones independientes.'}
    for i, label in enumerate(labels):
        item = dict(label) if isinstance(label, dict) else {'id': str(label)}
        output['by_class'].append({**item, **row(y == i, None if mask is None else mask & (pred == i))})
    return output
