"""Auditoría e intercambio local de datasets. Sin pickle ni extracción de rutas ZIP."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import uuid
import zipfile
import numpy as np
from sign_languages import dataset_dir, labels_for, language_info, pipeline_spec, register_label, clean_sign_name
from web_dataset import load_sequence, save_sequence, sequence_metadata, validate_identifier
from capture_quality import sequence_quality

# Objetivo de tomas originales disponibles para aprender después de reservar
# grupos completos. No es un mínimo obligatorio para iniciar el entrenamiento.
TRAIN_SUPPORT_TARGET = 15

MAX_ARCHIVE = 32 * 1024 * 1024
MAX_EXPANDED = 128 * 1024 * 1024
_CACHE = {}


# Calcula la huella de una secuencia de puntos para identificar copias exactas.
def digest(points):
    return hashlib.sha256(np.asarray(points,dtype=np.float32).tobytes()).hexdigest()


# Inspecciona los archivos de un perfil y describe validez, calidad, duplicados y procedencia.
def sample_rows(language):
    root = dataset_dir(language)
    files = sorted(root.glob('*/*.npz'))
    stamp = tuple((str(p), p.stat().st_mtime_ns,p.stat().st_size) for p in files)
    cache_key = str(root)
    if cache_key in _CACHE and _CACHE[cache_key][0] == stamp:
        return _CACHE[cache_key][1]
    rows, seen = [], {}
    for label in labels_for(language):
        for path in sorted((root / label['id']).glob('*.npz')):
            row = {'id':path.stem,'label':label['id'],'text':label['text'], 'bytes':path.stat().st_size,
                   'created_at':path.stat().st_mtime,'valid':False,'duplicate_of':None,'conflict':False}
            try:
                points = load_sequence(path)
                meta = sequence_metadata(path)
                fingerprint = digest(points)
                quality = sequence_quality(points)
                previous = seen.get(fingerprint)
                row.update(valid=True,sha256=fingerprint,metadata=meta,quality=quality,
                           duplicate_of=previous['id'] if previous else None,
                           conflict=bool(previous and previous['label'] != label['id']))
                if previous:
                    previous['conflict'] |= row['conflict']
                else:
                    seen[fingerprint] = row
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
                row['error'] = str(error)
            rows.append(row)
    _CACHE[cache_key] = stamp, rows
    return rows


# Resume problemas del dataset para mostrarlos en la revisión de grabaciones.
def audit_dataset(language):
    rows = sample_rows(language)
    return {'language':language,'samples':rows,'summary':{
        'total':len(rows),'valid':sum(r['valid'] for r in rows),
        'duplicates':sum(bool(r['duplicate_of']) for r in rows),
        'conflicts':sum(r['conflict'] for r in rows),
        'low_quality':sum(r.get('quality',{}).get('low_quality',False) for r in rows),
        'without_provenance':sum(not r.get('metadata',{}).get('session_id') for r in rows)}}


# Construye un ZIP con manifiesto y ejemplos válidos de una sola lengua.
def export_dataset(language):
    language_info(language)
    output = io.BytesIO()
    manifest = {'format':'sena-dataset-v1','language':language,'pipeline':pipeline_spec(),'samples':[]}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for row in sample_rows(language):
            if not row['valid']:
                continue
            validate_identifier(row['id'])
            filename = f"samples/{row['label']}/{row['id']}.npz"
            archive.writestr(filename,(dataset_dir(language)/row['label']/f"{row['id']}.npz").read_bytes())
            manifest['samples'].append({'file':filename,'name':row['text'],'label_id':row['label']})
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False))
    if len(output.getvalue()) > MAX_ARCHIVE:
        raise ValueError('La exportación supera 32 MB. Divide el dataset por lotes desde sus carpetas.')
    return {'filename':f'sena-{language}.zip','data':base64.b64encode(output.getvalue()).decode(),
            'samples':len(manifest['samples'])}


# Valida el ZIP completo, sus rutas y muestras antes de incorporar ejemplos compatibles.
def import_dataset(language, encoded):
    language_info(language)
    if not isinstance(encoded,str) or len(encoded) > MAX_ARCHIVE * 1.4:
        raise ValueError('El archivo supera el límite de 32 MB.')
    try:
        content = base64.b64decode(encoded,validate=True)
        if len(content) > MAX_ARCHIVE:
            raise ValueError('El archivo supera 32 MB.')
        archive = zipfile.ZipFile(io.BytesIO(content))
        infos = archive.infolist()
        if len(infos) > 2001 or sum(i.file_size for i in infos) > MAX_EXPANDED or len({i.filename for i in infos}) != len(infos):
            raise ValueError('Archivo demasiado grande o con nombres repetidos.')
        for entry in infos:
            if entry.file_size > 2 * 1024 * 1024 or entry.filename.startswith('/') or '..' in Path(entry.filename).parts:
                raise ValueError('El archivo contiene rutas o tamaños no permitidos.')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['format'] != 'sena-dataset-v1' or manifest['language'] != language or manifest['pipeline'] != pipeline_spec():
            raise ValueError('El paquete pertenece a otra lengua o a una configuración incompatible.')
        entries = manifest['samples']
        if not isinstance(entries,list) or not entries or len(entries)>2000:
            raise ValueError('El paquete debe contener de 1 a 2000 ejemplos.')
        known = {r.get('sha256'):r['text'].lower() for r in sample_rows(language) if r['valid']}
        pending, skipped = [], 0
        used_files=set()
        for entry in entries:
            name = clean_sign_name(entry['name'])
            filename=entry['file']
            if not isinstance(filename,str) or not filename.startswith('samples/') or filename in used_files:
                raise ValueError('Referencia de ejemplo repetida o inválida.')
            used_files.add(filename)
            raw = archive.read(filename)
            with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                if sum(i.file_size for i in inner.infolist()) > 2*1024*1024:
                    raise ValueError('Ejemplo comprimido demasiado grande.')
            with np.load(io.BytesIO(raw),allow_pickle=False) as record:
                from web_dataset import validate_sequence
                if json.loads(str(record['pipeline'].item())) != pipeline_spec():
                    raise ValueError('Un ejemplo tiene configuración incompatible.')
                points=validate_sequence(record['points'])
                metadata=json.loads(str(record['metadata'].item())) if 'metadata' in record else {}
                if not isinstance(metadata,dict):
                    raise ValueError('Metadatos de ejemplo inválidos.')
                metadata={k:str(metadata[k])[:80] for k in ('session_id','batch_id','participant','camera') if k in metadata}
            fingerprint=digest(points)
            if fingerprint in known:
                if known[fingerprint] != name.lower():
                    raise ValueError('El mismo ejemplo aparece con nombres distintos. Corrige las etiquetas antes de importar.')
                skipped+=1
                continue
            known[fingerprint]=name.lower()
            pending.append((name,points.copy(),metadata,entry.get('label_id')))
    except (KeyError,TypeError,zipfile.BadZipFile,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError('Paquete de Seña inválido o incompleto.') from exc
    vocabulary=dataset_dir(language).parent.parent/'vocabulary'/f'{language}.json'
    previous=vocabulary.read_bytes() if vocabulary.exists() else None
    created=[]
    try:
        for name,points,metadata,preferred in pending:
            label=next((item for item in labels_for(language) if item['id']==preferred and item['text']==name),None) or register_label(language,name)
            filename=save_sequence(language,label['id'],points,metadata=metadata)
            created.append(dataset_dir(language)/label['id']/filename)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        if previous is None:
            vocabulary.unlink(missing_ok=True)
        else:
            vocabulary.write_bytes(previous)
        raise
    return {'imported':len(created),'skipped_duplicates':skipped}


# Agrupa ejemplos por personas o sesiones y vincula copias para impedir fugas entre conjuntos.
def training_provenance(language, raw, targets, labels):
    rows=[r for r in sample_rows(language) if r['valid']]
    lookup={r.get('sha256'):r for r in rows}
    selected_ids={l['id'] for l in labels}
    rows=[r for r in rows if r['label'] in selected_ids]
    people_by_class=[{r.get('metadata',{}).get('participant') for r in rows if r['label']==label['id'] and r.get('metadata',{}).get('participant') not in (None, '', 'user:local')} for label in labels]
    by_person=(all(len(people)>=3 for people in people_by_class)
               and all(r.get('metadata',{}).get('participant') not in (None, '', 'user:local') for r in rows))
    parent={}
    # Encuentra el representante de un grupo y comprime sus enlaces de equivalencia.
    def root(value):
        parent.setdefault(value,value)
        while parent[value]!=value:
            parent[value]=parent[parent[value]]
            value=parent[value]
        return value
    # Deriva la identidad de agrupación a partir de participante, sesión o huella histórica.
    def group_for(row):
        meta=row.get('metadata',{})
        return ('person:'+meta['participant'] if by_person and meta.get('participant') else
                'session:'+meta['session_id'] if meta.get('session_id') else 'legacy:'+row['sha256'])
    fingerprints={}
    for row in rows:
        group=root(group_for(row))
        fingerprint=row['sha256']
        if fingerprint in fingerprints:
            parent[root(group)]=root(fingerprints[fingerprint])
        else:
            fingerprints[fingerprint]=group
    groups=[]; legacy=0
    for points in raw:
        fingerprint=digest(points)
        row=lookup.get(fingerprint,{'sha256':fingerprint})
        meta=row.get('metadata',{})
        legacy+=int(not meta.get('session_id') and not (by_person and meta.get('participant')))
        groups.append(root(group_for(row)))
    if any(r['conflict'] for r in rows):
        raise ValueError('Hay ejemplos idénticos con etiquetas distintas. Revisa el dataset.')
    return np.array(groups), {'legacy_samples':legacy, 'group_count':len(set(groups)),
        'split_unit':'person' if by_person else 'session', 'person_independent':by_person and not legacy,
        'note':'Históricos sin sesión: evaluación exploratoria, no demuestra generalización a personas nuevas.' if legacy else ('Separación por personas.' if by_person else 'Separación por sesiones; hacen falta tres personas por clase para evaluar personas nuevas.')}


# Busca un reparto por grupos completos donde todas las clases estén representadas en cada conjunto.
def grouped_split(targets, groups, seed=42):
    """Grupos enteros en cada conjunto; no divide tandas ni participantes."""
    rng=np.random.default_rng(seed)
    unique=np.unique(groups)
    if len(unique)<3:
        raise ValueError('Necesitas al menos tres sesiones o participantes independientes para separar entrenamiento, validación y prueba.')
    n=max(1,round(len(unique)*.15))
    best=None; score=float('inf')
    all_labels=set(targets.tolist())
    classes = np.array(sorted(all_labels))
    totals = np.array([np.sum(targets == label) for label in classes])
    for _ in range(1200):
        shuffled=rng.permutation(unique)
        parts=[np.flatnonzero(np.isin(groups,g)) for g in (shuffled[2*n:],shuffled[n:2*n],shuffled[:n])]
        if any(set(targets[p].tolist())!=all_labels for p in parts):
            continue
        current=sum(abs(len(part)/len(targets)-fraction) for part,fraction in zip(parts,(.7,.15,.15)))
        counts = np.array([[np.sum(targets[part] == label) for label in classes] for part in parts])
        if np.any(counts[0] < np.maximum(2, np.floor(totals * .4))):
            continue
        # Overall balance alone can leave one letter with a single validation
        # example. Balance each class too, while keeping sessions indivisible.
        current += 2 * np.mean(np.abs(counts / totals - np.array([.7,.15,.15])[:,None]))
        current += 2 * np.maximum(3-counts[1],0).sum() + np.maximum(3-counts[2],0).sum()
        if current<score:
            best,score=parts,current
    if best is None:
        raise ValueError('No se pueden separar las clases sin mezclar sesiones/personas. Graba cada seña en al menos tres grupos independientes.')
    return tuple(best)


# Devuelve metadatos y curvas escalares de movimiento y calidad para revisar una toma.
def inspect_sample(language, identifier):
    validate_identifier(identifier)
    row=next((r for r in sample_rows(language) if r['id']==identifier),None)
    if not row or not row['valid']:
        raise ValueError('La toma no existe o está incompleta.')
    points=load_sequence(dataset_dir(language)/row['label']/f'{identifier}.npz')
    from capture_quality import landmark_quality
    from recognition_features import regions
    hands=regions(points)[-2:]
    trajectory=np.concatenate([h[:,0,:2] for h in hands],axis=1)
    motion=np.linalg.norm(np.diff(trajectory,axis=0),axis=1)
    return {'sample':row,'timeline':[{'epoch':i+1,'motion':float(motion[i-1]) if i else 0,
                                     'quality':float(landmark_quality(frame)['score'])} for i,frame in enumerate(points)]}


# Conserva el destino histórico de grupos y huellas al ampliar el vocabulario.
def persistent_split(targets, groups, path, seed=42, fingerprints=None, label_names=None):
    """Freeze assignments across vocabulary growth; never recycle a test group."""
    path=Path(path)
    if not path.exists():
        parts=grouped_split(targets,groups,seed)
        mapping={str(g):name for name,idx in zip(('train','validation','test'),parts) for g in set(groups[idx])}
    else:
        stored=json.loads(path.read_text(encoding='utf-8'))
        mapping=stored['groups']
        if any(v not in ('train','validation','test') for v in mapping.values()):raise ValueError('Partición persistente inválida.')
        # A participant may replace several session groups later. Refuse to
        # regroup previously reserved samples across partitions.
        if fingerprints is not None:
            samples=stored.get('samples',{})
            for group in set(groups):
                locked={samples[h] for h,g in zip(fingerprints,groups) if g==group and h in samples}
                if group in mapping:locked.add(mapping[group])
                if len(locked)>1:
                    raise ValueError('Una persona/grupo combina particiones históricas. Reúne una reserva nueva independiente; no se recicla la prueba.')
                if locked:mapping[group]=locked.pop()
        new=sorted(set(groups)-set(mapping))
        missing = []
        roles = {'train', 'validation', 'test'}
        for c in sorted(set(targets.tolist())):
            present = set(groups[targets == c])
            assigned = {mapping[g] for g in present if g in mapping}
            free = present - set(mapping)
            needed = len(roles - assigned) - len(free)
            if needed > 0:
                name = label_names[int(c)] if label_names is not None else str(c)
                missing.append(f'«{name}»: {len(present)} sesiones; faltan al menos {needed}')
        if missing:
            raise ValueError('Faltan sesiones independientes. ' + '; '.join(missing)
                             + '. En Dataset, apaga y vuelve a activar la cámara para iniciar otra sesión. '
                             'Graba nuevas tomas completas de esas señas (recomendamos al menos 10 por sesión) '
                             'y guárdalas con la misma etiqueta. La reserva histórica se conserva.')
        rng = np.random.default_rng(seed)
        best, score = None, None
        classes = set(targets.tolist())
        for _ in range(1600):
            trial=dict(mapping)
            trial.update(zip(new,rng.choice(['train','validation','test'],len(new),p=[.7,.15,.15])))
            candidate=tuple(np.flatnonzero([trial[g]==name for g in groups]) for name in ('train','validation','test'))
            if any(set(targets[idx].tolist())!=classes for idx in candidate):continue
            counts=np.array([[np.sum(targets[idx]==c) for c in sorted(classes)] for idx in candidate])
            if np.any(counts[0]<2):continue
            # Prioriza el soporte de las señas con pocas tomas para aprender,
            # manteniendo la cobertura de validación y prueba.
            training_shortfall = np.maximum(TRAIN_SUPPORT_TARGET - counts[0], 0).sum()
            evaluation_shortfall = np.maximum(3 - counts[1:], 0).sum()
            fractions = counts / counts.sum(axis=0)
            balance = float(np.mean(abs(fractions - np.array([.7, .15, .15])[:, None])))
            cost = (evaluation_shortfall, training_shortfall, balance)
            if best is None or cost < score:
                best, score = (trial, candidate), cost
        if best is None and new:
            # La búsqueda aleatoria puede omitir un reparto válido al añadir
            # muchas clases. Resolver la cobertura sin mover grupos históricos.
            from scipy.optimize import milp, Bounds, LinearConstraint
            names = ('train', 'validation', 'test')
            matrix, lower, upper = [], [], []
            for j in range(len(new)):
                row = np.zeros(len(new)*3); row[j*3:j*3+3] = 1
                matrix.append(row); lower.append(1); upper.append(1)
            for c in sorted(classes):
                for role, name in enumerate(names):
                    row = np.zeros(len(new)*3)
                    for j, group in enumerate(new):
                        row[j*3+role] = np.sum((groups == group) & (targets == c))
                    reserved = sum(targets[i] == c and mapping.get(g) == name
                                   for i, g in enumerate(groups))
                    matrix.append(row)
                    lower.append(max(0, (2 if role == 0 else 1)-reserved))
                    upper.append(np.inf)
            solution = milp(np.tile([0., 1., 1.], len(new)),
                            integrality=np.ones(len(new)*3), bounds=Bounds(0, 1),
                            constraints=LinearConstraint(np.array(matrix), lower, upper),
                            options={'time_limit': 5.0})
            if solution.x is not None:
                trial = dict(mapping)
                trial.update({g:names[int(np.argmax(solution.x[j*3:j*3+3]))]
                              for j,g in enumerate(new)})
                candidate = tuple(np.flatnonzero([trial[g] == name for g in groups]) for name in names)
                if (all(set(targets[idx].tolist()) == classes for idx in candidate)
                        and all(np.sum(targets[candidate[0]] == c) >= 2 for c in classes)):
                    best = (trial, candidate)
        if best is None:
            raise ValueError('No se encontró un reparto compatible de las sesiones seleccionadas. '
                             'Las sesiones compartidas entre clases deben permanecer juntas. '
                             'Graba sesiones adicionales de las clases nuevas; tus tomas y modelos se conservan.')
        mapping,parts=best
    path.parent.mkdir(parents=True,exist_ok=True)
    samples = stored.get('samples', {}) if path.exists() else {}
    if fingerprints is not None:samples.update({h:mapping[g] for h,g in zip(fingerprints,groups)})
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps({'version':1,'groups':mapping,'samples':samples},ensure_ascii=False), encoding='utf-8');os.replace(temp,path)
    return tuple(parts)
