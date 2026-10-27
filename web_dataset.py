"""Persistencia y validación de secuencias web completas, sin imágenes."""
import json
import os
import uuid
from datetime import datetime, timezone
import zipfile
from pathlib import Path
import numpy as np
from config import FEATURE_DIM, SEQUENCE_LENGTH
from sign_languages import dataset_dir, labels_for, validate_label, pipeline_spec, register_label

def validate_sequence(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(points).all():
        raise ValueError('La secuencia tiene dimensiones o valores inválidos.')
    if np.count_nonzero(np.any(points != 0, axis=1)) < SEQUENCE_LENGTH * 0.8:
        raise ValueError('Faltan detecciones en demasiados fotogramas. Repite la captura.')
    return points

def save_sequence(language, label, points, sample_id=None):
    validate_label(language, label)
    points = validate_sequence(points)
    path = dataset_dir(language) / label
    path.mkdir(parents=True, exist_ok=True)
    identifier = sample_id or uuid.uuid4().hex
    validate_identifier(identifier)
    target = path / f'{identifier}.npz'
    temp = target.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, points=points, pipeline=json.dumps(pipeline_spec()))
    os.replace(temp, target)
    return target.name

def load_sequence(path):
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data['pipeline'].item())) != pipeline_spec():
            raise ValueError('La secuencia pertenece a otra configuración de captura.')
        return validate_sequence(data['points'])

_INVENTORY_CACHE = {}


def inventory(language):
    # Leer los arrays solo si cambia algún archivo. Consultar el estado no debe
    # interrumpir el ritmo de captura al crecer los siete datasets.
    root = dataset_dir(language)
    files = sorted(root.glob('*/*.npz'))
    fingerprint = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    key = (str(root), json.dumps(pipeline_spec(), sort_keys=True))
    fingerprint += (json.dumps(labels_for(language), ensure_ascii=False, sort_keys=True),)
    cached = _INVENTORY_CACHE.get(key)
    if cached and cached[0] == fingerprint:
        return cached[1]
    result = []
    for label in labels_for(language):
        valid = invalid = 0
        for path in sorted((dataset_dir(language) / label['id']).glob('*.npz')):
            try:
                load_sequence(path)
                valid += 1
            except (ValueError, KeyError, OSError, zipfile.BadZipFile):
                invalid += 1
        result.append({**label, 'samples': valid, 'invalid': invalid})
    _INVENTORY_CACHE[key] = (fingerprint, result)
    return result

def archive_label(language, label):
    """Retira de forma atómica una clase del dataset y conserva sus archivos."""
    validate_label(language, label)
    source = dataset_dir(language) / label
    if source.is_symlink():
        raise ValueError('La carpeta de ejemplos no puede ser un enlace simbólico.')
    samples = list(source.glob('*.npz'))
    if not samples:
        raise ValueError('Esta palabra o letra no tiene grabaciones guardadas para borrar.')
    identifier = uuid.uuid4().hex
    archive = dataset_dir(language).parent.parent / 'trash' / language / identifier
    archive.mkdir(parents=True, exist_ok=False)
    # Escribir el registro primero: si falla, el dataset aún no se ha modificado.
    (archive / 'deletion.json').write_text(json.dumps({
        'id': identifier, 'language': language, 'label': label, 'samples': len(samples),
        'deleted_at': datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    os.rename(source, archive / label)
    _INVENTORY_CACHE.clear()
    return {'deleted': len(samples), 'archive_id': identifier}


def read_dataset(language, minimum, label_ids=None):
    if label_ids is not None:
        if not isinstance(label_ids, list) or len(label_ids) < 2 or any(not isinstance(label, str) for label in label_ids) or len(set(label_ids)) != len(label_ids):
            raise ValueError('Selecciona al menos dos señas diferentes para entrenar.')
        for label in label_ids:
            validate_label(language, label)
    arrays, targets, labels = [], [], []
    for item in inventory(language):
        if label_ids is not None and item['id'] not in label_ids:
            continue
        if item['samples'] == 0 and label_ids is None:
            continue
        if item['samples'] < minimum:
            raise ValueError(f"{item['text']}: requiere al menos {minimum} ejemplos completos; hay {item['samples']}.")
        class_id = len(labels)
        labels.append({k: item[k] for k in ('id', 'text', 'meaning')})
        for path in sorted((dataset_dir(language) / item['id']).glob('*.npz')):
            try:
                arrays.append(load_sequence(path))
                targets.append(class_id)
            except (ValueError, KeyError, OSError, zipfile.BadZipFile):
                continue
    if len(labels) < 2:
        raise ValueError(f'Graba al menos dos señas diferentes, con {minimum} ejemplos completos por seña.')
    return np.stack(arrays), np.asarray(targets, dtype=np.int32), labels

def split_per_class(y, seed=42):
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        # Con 5–6 ejemplos, reservar uno por conjunto deja al menos 3 para entrenar.
        holdout = 1 if len(indices) < 7 else max(2, round(len(indices) * .15))
        if len(indices) < 5 or len(indices) - 2 * holdout < 2:
            raise ValueError('No hay suficientes ejemplos para separar entrenamiento, validación y prueba.')
        test.extend(indices[:holdout])
        validation.extend(indices[holdout:2 * holdout])
        train.extend(indices[2 * holdout:])
    return tuple(np.array(part, dtype=int) for part in (train, validation, test))


def validate_identifier(identifier):
    if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('Identificador de grabación inválido.')
    return identifier


def draft_dir(language):
    # Derivar del directorio validado evita usar un idioma como ruta arbitraria.
    return dataset_dir(language).parent.parent / 'drafts' / language


def save_draft(language, points):
    points = validate_sequence(points)
    folder = draft_dir(language)
    folder.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    path = folder / f'{identifier}.npz'
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, points=points, pipeline=json.dumps(pipeline_spec()))
    os.replace(temp, path)
    return identifier


def pending_drafts(language):
    return [{'id': p.stem, 'created_at': p.stat().st_mtime} for p in sorted(
        draft_dir(language).glob('*.npz'), key=lambda p: p.stat().st_mtime)]


def commit_draft(language, identifier, name):
    validate_identifier(identifier)
    folder = draft_dir(language)
    receipt = folder / f'{identifier}.saved.json'
    # Reintentar un guardado cuya respuesta se perdió no duplica el ejemplo.
    if receipt.exists():
        (folder / f'{identifier}.npz').unlink(missing_ok=True)
        return json.loads(receipt.read_text(encoding='utf-8'))
    path = folder / f'{identifier}.npz'
    if not path.is_file():
        raise ValueError('La grabación no existe en esta lengua. Vuelve a grabar la seña.')
    points = load_sequence(path)
    label = register_label(language, name)
    sample = save_sequence(language, label['id'], points, sample_id=identifier)
    result = {'label': label, 'sample': sample, 'language': language}
    temp = receipt.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, receipt)
    path.unlink()
    return result


def discard_draft(language, identifier):
    validate_identifier(identifier)
    path = draft_dir(language) / f'{identifier}.npz'
    if path.exists():
        # Conservar una copia recuperable, fuera de los ejemplos de entrenamiento.
        discarded = path.parent / 'discarded'
        discarded.mkdir(exist_ok=True)
        os.replace(path, discarded / path.name)
