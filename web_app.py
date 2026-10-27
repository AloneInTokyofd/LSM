"""Página local para recolectar, entrenar y probar siete lenguas de señas.
Ejecutar: .venv/bin/python web_app.py — abrir http://127.0.0.1:8765
"""
from __future__ import annotations
import argparse
import base64
from collections import deque
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import uuid

import numpy as np
from config import BASE_DIR, FEATURE_DIM, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, STABILITY_FRAMES, COOLDOWN_SECONDS
from sign_languages import LANGUAGES, WEB_DATA, MIN_SAMPLES, RECOMMENDED_SAMPLES, CAPTURE_FPS, language_info, labels_for, validate_label, model_dir, pipeline_spec
from recognition_features import prepare_sequence, feature_dimension, hand_guard_accept
from web_dataset import inventory, read_dataset, save_draft, pending_drafts, commit_draft, discard_draft, archive_label
from admin_auth import AdminGuard
from rig_overlay import rig_payload

from alphabets import alphabet_status, add_alphabet, add_all_alphabets

ROOT = Path(BASE_DIR)
TOKEN = secrets.token_urlsafe(32)


def model_metadata(language):
    directory = model_dir(language)
    manifest = directory / 'current.json'
    if not manifest.exists():
        return None
    selection = json.loads(manifest.read_text())
    if selection.get('requires_retraining'):
        return None
    version = selection['version']
    if not isinstance(version, str) or len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
        raise ValueError('La versión del modelo es inválida.')
    path = directory / version
    meta = json.loads((path / 'metadata.json').read_text())
    if meta['language'] != language or meta['pipeline'] != pipeline_spec():
        raise ValueError('El modelo no corresponde a esta lengua o a la configuración de captura actual.')
    feature_dimension(meta.get('preprocessing'))
    if not (path / 'model.keras').is_file() or len(meta['labels']) < 2:
        raise ValueError('Faltan el modelo o sus etiquetas.')
    return path, meta


class Session:
    def __init__(self, language):
        self.language = language
        self.holistic = None
        self.touched = time.monotonic()
        self.last_frame = 0.0
        self.capture = None
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=SEQUENCE_LENGTH)
        self.probabilities = deque(maxlen=5)
        self.stable_label = None
        self.stable_count = 0
        self.last_confirmed = None
        self.cooldown = 0
        self.no_hands = 0

    def close(self):
        if self.holistic is not None:
            self.holistic.close()


class Runtime:
    def __init__(self):
        self.lock = threading.RLock()
        self.sessions = {}
        self.models = {}
        self.process = None
        self.job_id = None
        self.admin = AdminGuard(WEB_DATA / 'admin' / 'auth.json')

    def invalidate_label_model(self, language, label):
        manifest = model_dir(language) / 'current.json'
        if not manifest.is_file():
            return False
        selection = json.loads(manifest.read_text())
        version = selection['version']
        if not isinstance(version, str) or len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
            raise ValueError('La versión del modelo es inválida.')
        metadata = json.loads((manifest.parent / version / 'metadata.json').read_text())
        if label not in {item['id'] for item in metadata['labels']}:
            return False
        selection['requires_retraining'] = True
        selection['reason'] = 'Se borraron ejemplos de una seña entrenada. Reentrena para actualizar el vocabulario; el modelo anterior está desactivado.'
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, manifest)
        self.models.pop(language, None)
        for item in self.sessions.values():
            if item.language == language:
                item.reset()
        return True

    def delete_recordings(self, payload):
        self.admin.verify(payload.get('password'))
        if payload.get('confirmed') is not True:
            raise ValueError('Confirma que quieres borrar todos los ejemplos de esta palabra o letra.')
        if self.training() or any(item.capture for item in self.sessions.values()):
            raise ValueError('Detén las grabaciones y espera a que termine el entrenamiento antes de borrar ejemplos.')
        language, label = payload.get('language'), payload.get('label')
        # Desactivar primero: ni siquiera un fallo de archivo debe dejar
        # disponible una clase cuya eliminación ya se ha autorizado.
        validate_label(language, label)
        invalidated = self.invalidate_label_model(language, label)
        result = archive_label(language, label)
        result['model_invalidated'] = invalidated
        for item in self.sessions.values():
            if item.language == language:
                item.reset()
        return result

    def new_session(self, language):
        language_info(language)
        now = time.monotonic()
        for key, item in list(self.sessions.items()):
            if now - item.touched > 600:
                item.close()
                del self.sessions[key]
        if len(self.sessions) >= 4:
            raise ValueError('Ya hay cuatro sesiones abiertas. Detén una cámara y vuelve a intentarlo.')
        identifier = uuid.uuid4().hex
        self.sessions[identifier] = Session(language)
        return identifier

    def session(self, identifier):
        if identifier not in self.sessions:
            raise ValueError('La sesión terminó. Detén y vuelve a activar la cámara.')
        item = self.sessions[identifier]
        item.touched = time.monotonic()
        return item

    def start_capture(self, identifier, consecutive=False):
        if not isinstance(consecutive, bool):
            raise ValueError('La opción de grabación consecutiva no es válida.')
        item = self.session(identifier)
        pending = pending_drafts(item.language)
        if pending and not consecutive:
            raise ValueError('Pon nombre o descarta la grabación pendiente antes de hacer otra, o usa una tanda consecutiva.')
        if len(pending) >= 100:
            raise ValueError('Hay 100 tomas pendientes en esta lengua. Nombra y guarda algunas antes de continuar.')
        if self.training() or item.capture:
            raise ValueError('Termina la operación actual antes de grabar otra secuencia.')
        item.reset()
        item.capture = {'points': [], 'hands': 0}
        return {'capturing': True}

    def job(self):
        if not self.job_id:
            return None
        path = WEB_DATA / 'jobs' / f'{self.job_id}.json'
        state = {'id': self.job_id, 'state': 'running', 'message': 'Preparando entrenamiento…', 'epoch': 0}
        if path.exists():
            state = json.loads(path.read_text())
        if self.process is not None and self.process.poll() is not None and state['state'] == 'running':
            state.update(state='failed', message='El proceso terminó antes de completar el modelo. Revisa web_data/jobs/' + self.job_id + '.log')
        if 'result' in state:
            state['result'] = {k: v for k, v in state['result'].items() if k != 'hand_guard'}
        return state

    def training(self):
        return self.process is not None and self.process.poll() is None

    def state(self):
        profiles = []
        for info in LANGUAGES:
            rows = inventory(info['id'])
            metadata = None
            error = None
            try:
                result = model_metadata(info['id'])
                metadata = {k: v for k, v in result[1].items() if k != 'hand_guard'} if result else None
                if not result:
                    manifest = model_dir(info['id']) / 'current.json'
                    if manifest.is_file():
                        selection = json.loads(manifest.read_text())
                        if selection.get('requires_retraining'):
                            error = selection.get('reason', 'Reentrena el modelo tras borrar ejemplos.')
            except (ValueError, KeyError, OSError) as exc:
                error = str(exc)
            profiles.append({**info, 'labels': rows, 'model': metadata, 'model_error': error, 'drafts': pending_drafts(info['id']), 'alphabet': alphabet_status(info['id'], rows, metadata)})
        old_model = ROOT / 'models' / 'lsm_model.h5'
        return {'profiles': profiles, 'minimum_samples': MIN_SAMPLES, 'recommended_samples': RECOMMENDED_SAMPLES, 'sequence_length': SEQUENCE_LENGTH,
                'target_fps': CAPTURE_FPS, 'token': TOKEN, 'job': self.job(),
                'legacy_model': old_model.is_file()}

    def load_model(self, language):
        bundle = model_metadata(language)
        if not bundle:
            return None
        directory, meta = bundle
        cached = self.models.get(language)
        if not cached or cached[0] != directory:
            import tensorflow as tf
            model = tf.keras.models.load_model(directory / 'model.keras', compile=False, safe_mode=True)
            if tuple(model.input_shape[1:]) != (SEQUENCE_LENGTH, feature_dimension(meta.get('preprocessing'))) or model.output_shape[-1] != len(meta['labels']):
                raise ValueError('Las dimensiones del modelo no coinciden con sus metadatos.')
            self.models[language] = directory, model, meta
        return self.models[language][1:]

    def frame(self, payload):
        if self.training():
            raise ValueError('Espera a que termine el entrenamiento antes de usar la cámara.')
        session = self.session(payload.get('session'))
        import cv2
        from utils import mediapipe_detection, extract_keypoints, hands_detected
        encoded = payload.get('image', '')
        if not isinstance(encoded, str) or len(encoded) > 1500000:
            raise ValueError('Fotograma demasiado grande.')
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('Fotograma no válido.') from exc
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[0] > 1080 or image.shape[1] > 1920:
            raise ValueError('Envía una imagen JPEG de hasta 1920 × 1080.')
        if session.holistic is None:
            import mediapipe as mp
            session.holistic = mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=.5, min_tracking_confidence=.5)
        now = time.monotonic()
        gap = now - session.last_frame if session.last_frame else 0
        session.last_frame = now
        if gap > 1.0:
            session.reset()
            if session.capture:
                session.capture = None
                raise ValueError('La captura se interrumpió. Graba la secuencia de nuevo con una conexión estable.')
        _, results = mediapipe_detection(cv2.flip(image, 1), session.holistic)
        points = extract_keypoints(results)
        hands = hands_detected(results)
        output = {'hands': hands, 'buffer': len(session.frames), 'captured': 0, 'draft_id': None,
                  'candidate': None, 'confirmed': None, 'confidence': 0, 'stability': 0, 'rejection_reason': None}
        if payload.get('rig') is True:
            output['rig'] = rig_payload(points, image.shape[1], image.shape[0])
        if session.capture is not None:
            capture = session.capture
            capture['points'].append(points)
            capture['hands'] += int(hands)
            output['captured'] = len(capture['points'])
            if len(capture['points']) == SEQUENCE_LENGTH:
                session.capture = None
                if capture['hands'] < SEQUENCE_LENGTH * .8:
                    raise ValueError('Las manos no fueron visibles en suficientes fotogramas. Repite la seña.')
                output['draft_id'] = save_draft(session.language, capture['points'])
                session.reset()
            return output
        if payload.get('mode') != 'test':
            return output
        session.frames.append(points)
        output['buffer'] = len(session.frames)
        if not hands:
            session.probabilities.clear()
            session.stable_label, session.stable_count = None, 0
            session.no_hands += 1
            if session.no_hands >= 5:
                session.last_confirmed = None
                session.frames.clear()
            return output
        session.no_hands = 0
        bundle = self.load_model(session.language)
        if not bundle or len(session.frames) < SEQUENCE_LENGTH:
            return output
        model, metadata = bundle
        raw_sequence = np.stack(session.frames)
        sequence = prepare_sequence(raw_sequence, metadata.get('preprocessing'))
        probs = model(sequence[None, ...], training=False).numpy()[0]
        session.probabilities.append(probs)
        probs = np.mean(session.probabilities, axis=0)
        index = int(np.argmax(probs))
        label = metadata['labels'][index]
        confidence = float(probs[index])
        output.update(candidate=label, confidence=confidence)
        known_shape = hand_guard_accept(raw_sequence, index, metadata.get('hand_guard'))
        if not known_shape:
            output['rejection_reason'] = 'La forma de la mano se aleja de los ejemplos aprendidos. Repite la seña o añade ejemplos variados.'
        threshold = metadata.get('decision_threshold', CONFIDENCE_THRESHOLD)
        margin = float(np.sort(probs)[-1] - np.sort(probs)[-2])
        if not known_shape or confidence < threshold or margin < metadata.get('decision_margin', .15):
            session.stable_label, session.stable_count = None, 0
        elif session.stable_label == label['id']:
            session.stable_count += 1
        else:
            session.stable_label, session.stable_count = label['id'], 1
        output['stability'] = min(session.stable_count, STABILITY_FRAMES)
        if (session.stable_count >= STABILITY_FRAMES and now >= session.cooldown
                and label['id'] != session.last_confirmed):
            output['confirmed'] = label
            session.last_confirmed = label['id']
            session.cooldown = now + COOLDOWN_SECONDS
        return output

    def start_training(self, language, epochs, label_ids=None):
        language_info(language)
        if self.training():
            raise ValueError('Ya hay un entrenamiento en curso.')
        if not isinstance(epochs, int) or not 5 <= epochs <= 200:
            raise ValueError('Selecciona entre 5 y 200 épocas.')
        if pending_drafts(language):
            raise ValueError('Pon nombre o descarta las grabaciones pendientes antes de entrenar.')
        read_dataset(language, MIN_SAMPLES, label_ids)
        for session in self.sessions.values():
            if session.capture:
                raise ValueError('Termina o cancela la captura antes de entrenar.')
        self.job_id = uuid.uuid4().hex
        logs = WEB_DATA / 'jobs'
        logs.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        env['MPLCONFIGDIR'] = str(WEB_DATA / 'cache' / 'matplotlib')
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'
        selection_args = ['--labels', *label_ids] if label_ids is not None else []
        with (logs / f'{self.job_id}.log').open('w') as output:
            self.process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'web_train.py'),
                                             '--language', language, '--epochs', str(epochs), '--job-id', self.job_id, *selection_args],
                                            cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        return {'job_id': self.job_id}


RUNTIME = Runtime()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Evitar una línea por cada fotograma.
        if '/api/frame' not in str(args):
            super().log_message(format, *args)

    def send_data(self, status, data, content_type='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def trusted_host(self):
        host = self.headers.get('Host', '')
        return host in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    def do_GET(self):
        if not self.trusted_host():
            return self.send_data(403, {'error': 'Abre esta página desde localhost.'})
        route = urlsplit(self.path).path
        if route == '/api/state':
            with RUNTIME.lock:
                return self.send_data(200, RUNTIME.state())
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/static/rig.js': ('rig.js', 'text/javascript; charset=utf-8'),
                 '/static/continuous.js': ('continuous.js', 'text/javascript; charset=utf-8'),
                 '/static/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/static/styles.css': ('styles.css', 'text/css; charset=utf-8')}
        if route not in files:
            return self.send_data(404, {'error': 'Recurso no encontrado.'})
        filename, mime = files[route]
        self.send_data(200, (ROOT / 'static' / filename).read_bytes(), mime)

    def do_POST(self):
        origin = self.headers.get('Origin')
        expected = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        if not self.trusted_host() or (origin and origin not in expected) or self.headers.get('X-LSM-Token') != TOKEN:
            return self.send_data(403, {'error': 'Sesión no autorizada. Recarga la página.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 2000000:
                raise ValueError('Solicitud vacía o demasiado grande.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Solicitud no válida.')
            route = urlsplit(self.path).path
            with RUNTIME.lock:
                if route == '/api/session':
                    result = {'session': RUNTIME.new_session(payload.get('language'))}
                elif route == '/api/close':
                    item = RUNTIME.sessions.pop(payload.get('session'), None)
                    if item:
                        item.close()
                    result = {'closed': True}
                elif route == '/api/reset':
                    item = RUNTIME.session(payload.get('session'))
                    item.reset()
                    item.capture = None
                    result = {'reset': True}
                elif route == '/api/capture':
                    result = RUNTIME.start_capture(payload.get('session'), payload.get('consecutive', False))
                elif route == '/api/alphabet':
                    if RUNTIME.training():
                        raise ValueError('Espera a que termine el entrenamiento para añadir letras.')
                    if payload.get('scope') == 'all':
                        result = {'results': add_all_alphabets()}
                    else:
                        result = add_alphabet(payload.get('language'), payload.get('extras', False))
                elif route == '/api/capture/save':
                    if RUNTIME.training():
                        raise ValueError('Espera a que termine el entrenamiento para integrar otro ejemplo.')
                    result = commit_draft(payload.get('language'), payload.get('draft_id'), payload.get('name'))
                elif route == '/api/capture/discard':
                    discard_draft(payload.get('language'), payload.get('draft_id'))
                    result = {'discarded': True}
                elif route == '/api/frame':
                    result = RUNTIME.frame(payload)
                elif route == '/api/admin/delete-recordings':
                    result = RUNTIME.delete_recordings(payload)
                elif route == '/api/train':
                    result = RUNTIME.start_training(payload.get('language'), payload.get('epochs', 40), payload.get('labels'))
                else:
                    return self.send_data(404, {'error': 'Ruta no encontrada.'})
            self.send_data(200, result)
        except PermissionError as exc:
            self.send_data(403, {'error': str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            self.send_data(400, {'error': str(exc)})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.send_data(500, {'error': f'No se pudo completar la operación: {exc}'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    cache = WEB_DATA / 'cache' / 'matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'\nPágina de señas: http://127.0.0.1:{args.port}\nCtrl+C para cerrar.\n', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with RUNTIME.lock:
            for item in RUNTIME.sessions.values():
                item.close()
        if RUNTIME.training():
            print('El entrenamiento continúa en su proceso. Consulta web_data/jobs/.')


if __name__ == '__main__':
    main()
