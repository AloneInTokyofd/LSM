"""Operaciones locales del estudio, con aislamiento por lengua y controles administrativos."""
import json
import os
from pathlib import Path
import uuid
from datetime import datetime, timezone
import sign_languages as languages
from sign_languages import model_dir, dataset_dir, validate_label, language_info, pipeline_spec
from recognition_features import feature_dimension
from web_dataset import inventory, validate_identifier
from dataset_tools import audit_dataset, export_dataset, import_dataset, sample_rows, inspect_sample


def current_selection(root):
    try:
        selected = json.loads((root / 'current.json').read_text(encoding='utf-8'))
        return selected if isinstance(selected, dict) else {}
    except (OSError, ValueError):
        return {}


def compatible_metadata(language):
    """Lee una versión a la vez y descarta pipelines incompatibles."""
    for path in model_dir(language).glob('*/metadata.json'):
        try:
            version = path.parent.name
            validate_identifier(version)
            meta = json.loads(path.read_text(encoding='utf-8'))
            if meta['language'] != language or meta['pipeline'] != pipeline_spec():
                continue
            feature_dimension(meta.get('preprocessing'))
            yield version, meta, path.stat().st_mtime
        except (ValueError, KeyError, OSError):
            continue


# Lista versiones y diagnósticos para la vista de Entrenamiento.
def versions(language):
    selected = current_selection(model_dir(language))
    result = []
    for version, meta, modified in compatible_metadata(language):
        if meta.get('confusion_matrix') and not meta.get('test_evaluation'):
            from recognition_analysis import confusion_analysis
            meta['legacy_confusions'] = confusion_analysis(meta['confusion_matrix'], meta['labels'])
        summary = {k: v for k, v in meta.items() if k not in (
            'posture_vote', 'hand_guard', 'temporal_guard', 'split_hashes', 'split_groups'
        )}
        active = selected.get('version') == version and not selected.get('requires_retraining')
        result.append((modified, dict(summary, version=version, active=active)))
    return [row for _, row in sorted(result, key=lambda item: item[0], reverse=True)]


def load_inference_model(path):
    """Carga todos los pesos del modelo y da un diagnóstico de compatibilidad."""
    from importlib.metadata import version
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=True)
    except ValueError as error:
        message = str(error)
        if 'expected ' not in message or 'received 0 variables' not in message:
            raise
        python = r'.\.venv\Scripts\python.exe' if os.name == 'nt' else '.venv/bin/python'
        raise ValueError(
            f'No se pudieron cargar todos los pesos del modelo {path.parent.name} '
            f'con TensorFlow {tf.__version__} y Keras {version("keras")}. '
            f'En la carpeta LSM ejecuta {python} -m pip install -r requirements.txt '
            'y reinicia Seña. Si ya usas esas versiones, verifica que el archivo model.keras '
            'coincida con el del repositorio.'
        ) from error


# Valida vocabulario, archivos y dimensiones antes de cambiar atómicamente la versión activa.
def activate_version(language, version):
    validate_identifier(version)
    root=model_dir(language)
    metadata=json.loads((root/version/'metadata.json').read_text(encoding='utf-8'))
    if metadata['language']!=language or metadata['pipeline']!=pipeline_spec():
        raise ValueError('El modelo pertenece a otra lengua o configuración.')
    feature_dimension(metadata.get('preprocessing'))
    if not (root/version/'model.keras').is_file() or len(metadata['labels'])<2:
        raise ValueError('La versión está incompleta.')
    counts={r['id']:r['samples'] for r in inventory(language)}
    if any(counts.get(r['id'],0)<languages.MIN_SAMPLES for r in metadata['labels']):
        raise ValueError('Esta versión incluye señas eliminadas o con menos de 30 ejemplos. Reúne los ejemplos y reentrena.')
    # Comprobar dimensiones y serialización antes de tocar el puntero activo.
    model = load_inference_model(root / version / 'model.keras')
    expected=(pipeline_spec()['sequence_length'],feature_dimension(metadata.get('preprocessing')))
    if tuple(model.input_shape[1:])!=expected or model.output_shape[-1]!=len(metadata['labels']):
        raise ValueError('Dimensiones incompatibles con los metadatos.')
    temporary=root/'current.tmp'
    temporary.write_text(json.dumps({'version':version}))
    os.replace(temporary,root/'current.json')
    return {'activated':True,'version':version}


def activate_latest_version(language):
    """Selecciona el último modelo completo y compatible por fecha de entrenamiento."""
    root = model_dir(language)
    selection = current_selection(root)
    candidates = [
        (meta.get('created_at') or '', modified, version)
        for version, meta, modified in compatible_metadata(language)
        if meta.get('auto_activate') is not False and len(meta.get('labels', [])) >= 2
        and (root / version / 'model.keras').is_file()
    ]
    for _, _, version in sorted(candidates, reverse=True):
        # El modelo que ya está activo se validó al activarlo. Las operaciones
        # de borrado que lo invalidan marcan el manifiesto para reentrenar.
        if version == selection.get('version') and not selection.get('requires_retraining'):
            return version
        try:
            activate_version(language, version)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        return version
    return None


_ACTIVITY_CACHE = (None, [])


# Resume trabajos y operaciones recientes usando una caché basada en archivos modificados.
def recent_activity():
    global _ACTIVITY_CACHE
    jobs=sorted((languages.WEB_DATA/'jobs').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:8]
    deletions=sorted((languages.WEB_DATA/'trash').glob('*/*/deletion.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:8]
    files=jobs+deletions
    key=tuple((str(p),p.stat().st_mtime_ns) for p in files)
    if _ACTIVITY_CACHE[0]==key:
        return _ACTIVITY_CACHE[1]
    result=[]
    for p in files:
        try:
            record=json.loads(p.read_text(encoding='utf-8'))
            result.append({'kind':'training' if p in jobs else 'deletion','time':p.stat().st_mtime,
                           'language':record.get('language'),'state':record.get('state','archived'),
                           'message':record.get('message',f"{record.get('samples',1)} ejemplos retirados")})
        except (ValueError,OSError):
            continue
    result=sorted(result,key=lambda r:r['time'],reverse=True)[:10]
    _ACTIVITY_CACHE=key,result
    return result


# Enruta auditoría, importación, exportación y cambios administrativos dentro de una sola lengua.
def dataset_operation(runtime, route, payload):
    language=payload.get('language')
    language_info(language)
    if route=='/api/dataset/audit':
        return audit_dataset(language)
    if route=='/api/dataset/sample':
        return inspect_sample(language,payload.get('sample'))
    if route=='/api/dataset/export':
        return export_dataset(language)
    if route=='/api/dataset/import':
        if runtime.training() or any(s.capture for s in runtime.sessions.values()):
            raise ValueError('Termina las capturas y el entrenamiento antes de importar.')
        return import_dataset(language,payload.get('data'))
    if route not in ('/api/dataset/delete','/api/dataset/move'):
        raise ValueError('Operación de dataset no encontrada.')
    runtime.admin.verify(payload.get('password'))
    if payload.get('confirmed') is not True:
        raise ValueError('Confirma la modificación de ejemplos.')
    if runtime.training() or any(s.capture for s in runtime.sessions.values()):
        raise ValueError('Termina las capturas y el entrenamiento antes de modificar ejemplos.')
    selected=payload.get('samples')
    if not isinstance(selected,list) or not selected or len(selected)>2000:
        raise ValueError('Selecciona de 1 a 2000 ejemplos.')
    rows={r['id']:r for r in sample_rows(language)}
    if len(set(selected))!=len(selected) or any(not isinstance(key,str) or key not in rows for key in selected):
        raise ValueError('La selección de ejemplos no es válida.')
    if route.endswith('/move'):
        target=payload.get('target')
        validate_label(language,target)
        for key in selected:
            destination=dataset_dir(language)/target/f'{key}.npz'
            if destination.exists() and rows[key]['label']!=target:
                raise ValueError('Ya existe un ejemplo con ese identificador en el destino.')
    for key in selected:
        source=dataset_dir(language)/rows[key]['label']/f'{key}.npz'
        if not source.is_file() or source.is_symlink() or source.parent.is_symlink():
            raise ValueError('La selección cambió o contiene enlaces no permitidos. Vuelve a analizar el dataset.')
    # Invalidar cada clase afectada antes de modificar sus archivos.
    affected={rows[key]['label'] for key in selected}
    if route.endswith('/move'):
        affected.add(target)
    for label in affected:
        runtime.invalidate_label_model(language,label)
    receipt=languages.WEB_DATA/'trash'/language/uuid.uuid4().hex
    receipt.mkdir(parents=True,exist_ok=False)
    (receipt/'deletion.json').write_text(json.dumps({'language':language,'samples':len(selected),
        'operation':route.rsplit('/',1)[-1],'ids':selected,'original_labels':{key:rows[key]['label'] for key in selected},'deleted_at':datetime.now(timezone.utc).isoformat()},ensure_ascii=False))
    moved=[]
    try:
        for key in selected:
            row=rows[key];source=dataset_dir(language)/row['label']/f'{key}.npz'
            destination=(dataset_dir(language)/target if route.endswith('/move') else receipt/row['label'])/source.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            os.replace(source,destination)
            moved.append((source,destination))
    except Exception:
        for source,destination in reversed(moved):
            os.replace(destination,source)
        raise
    return {'changed':len(selected),'recoverable':route.endswith('/delete')}
