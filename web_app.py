"""Página local para recopilar ejemplos, entrenar y probar el modelo de LSM.
Ejecución local: .venv/bin/python web_app.py
Acceso HTTPS temporal: .venv/bin/python web_app.py --quick-tunnel
"""
from __future__ import annotations
import argparse
import base64
import hashlib
from importlib import import_module
from collections import deque
from functools import lru_cache
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import uuid

import numpy as np
from config import BASE_DIR, FEATURE_DIM, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, STABILITY_FRAMES, COOLDOWN_SECONDS
from sign_languages import LANGUAGES, WEB_DATA, MIN_SAMPLES, RECOMMENDED_SAMPLES, CAPTURE_FPS, language_info, labels_for, validate_label, model_dir, pipeline_spec
from recognition_features import prepare_sequence, feature_dimension, hand_guard_accept, temporal_guard_accept, augment_viewpoints
from web_dataset import inventory, stored_sample_counts, read_dataset, save_draft, pending_drafts, commit_draft, discard_draft, archive_label
from admin_auth import AdminGuard, TEAM_CONFIG
from accounts import Accounts
import team_access
from process_status import process_alive
from posture_vote import apply_vote
from recognition_analysis import temperature_probabilities
from recognition_decision import decision_details
from capture_quality import landmark_quality, image_quality, sequence_quality, hand_image_quality
from studio_services import versions, activate_latest_version, recent_activity, dataset_operation, load_inference_model

from alphabets import alphabet_status, add_alphabet, add_all_alphabets


QUICK_TUNNEL_URL = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com', re.IGNORECASE)


def start_quick_tunnel(port):
    """Abre un túnel HTTPS temporal a localhost y devuelve su proceso y URL."""
    executable = shutil.which('cloudflared')
    if not executable:
        raise RuntimeError('No encuentro cloudflared. Instálalo o inicia Seña solo en modo local.')
    process = subprocess.Popen(
        [executable, 'tunnel', '--url', f'http://127.0.0.1:{port}', '--no-autoupdate'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = queue.Queue()

    def read_output():
        for line in process.stdout:
            lines.put(line.rstrip())

    threading.Thread(target=read_output, daemon=True).start()
    deadline = time.monotonic() + 45
    recent = deque(maxlen=8)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('cloudflared terminó antes de crear el enlace. ' + ' | '.join(recent))
        try:
            line = lines.get(timeout=0.25)
        except queue.Empty:
            continue
        if line:
            print(f'[cloudflared] {line}', flush=True)
            recent.append(line)
            match = QUICK_TUNNEL_URL.search(line)
            if match:
                return process, match.group(0)
    process.terminate()
    raise RuntimeError('No se obtuvo un enlace temporal de Cloudflare en 45 segundos. ' + ' | '.join(recent))

ROOT = Path(BASE_DIR)
TOKEN = secrets.token_urlsafe(32)
ACCOUNTS = Accounts(WEB_DATA / "accounts" / "team.sqlite3")


# Valida el manifiesto activo, la lengua, el pipeline y los archivos del modelo antes de cargarlo.
@lru_cache(maxsize=4)
def public_model_metadata(path, modified, size):
    """Retiene solo los campos de la vista pública, sin guardar referencias pesadas."""
    full = json.loads(path.read_text(encoding='utf-8'))
    summary = {key: full[key] for key in (
        'language', 'pipeline', 'labels', 'created_at', 'preprocessing',
    ) if key in full}
    classes = (full.get('decision_policy') or {}).get('classes', {})
    summary['decision_policy'] = {'classes': {
        index: {'supported': row.get('supported')} for index, row in classes.items()
    }}
    return summary


def model_metadata(language, *, public=False):
    directory = model_dir(language)
    manifest = directory / 'current.json'
    if not manifest.exists():
        return None
    selection = json.loads(manifest.read_text(encoding='utf-8'))
    if selection.get('requires_retraining'):
        return None
    version = selection['version']
    if not isinstance(version, str) or len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
        raise ValueError('La versión del modelo es inválida.')
    path = directory / version
    metadata_path = path / 'metadata.json'
    if public:
        stat = metadata_path.stat()
        meta = public_model_metadata(metadata_path, stat.st_mtime_ns, stat.st_size)
    else:
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    if meta['language'] != language or meta['pipeline'] != pipeline_spec():
        raise ValueError('El modelo no corresponde a esta lengua o a la configuración de captura actual.')
    feature_dimension(meta.get('preprocessing'))
    if not (path / 'model.keras').is_file() or len(meta['labels']) < 2:
        raise ValueError('Faltan el modelo o sus etiquetas.')
    return path, meta


# Conserva detectores, buffers, procedencia y estado de una cámara; no representa una cuenta de usuario.
class Session:
    # Inicializa una cámara con sus detectores diferidos y contadores de seguimiento.
    def __init__(self, language):
        self.language = language
        self.owner_id = None
        self.holistic = None
        self.detail_regions = {}
        self.detail_detectors = {}
        self.detail_crops = {}
        self.inverted_detector = None
        self.inverted_until = 0.
        self.hand_detector = None
        self.snapshot_detector = None
        self.touched = time.monotonic()
        self.last_frame = 0.0
        self.last_hand_seen = 0.0
        self.hand_recovery_after = 0.0
        self.pose_recovery_after = 0.0
        self.pending_confirmation = None
        self.capture = None
        self.provenance = {}
        self.quality_gate = False
        self.reset()

    # Vacía la secuencia y sus votos de reconocimiento sin cerrar la cámara ni borrar tomas guardadas.
    def reset(self):
        self.frames = deque(maxlen=SEQUENCE_LENGTH + SEQUENCE_LENGTH//2)
        self.probabilities = deque(maxlen=5)
        self.stable_label = None
        self.stable_count = 0
        self.last_confirmed = None
        self.cooldown = 0
        self.no_hands = 0
        self.tracking_gap_start = None
        self.tracking_gap_frames = 0
        self.gesture_confirmed = False
        self.last_inference = 0
        self.last_model = None

    def update_recognition_tracking(self, hands, usable, now):
        """Tolera hasta dos cuadros malos sin añadir puntos antiguos al modelo."""
        self.no_hands = 0 if hands else self.no_hands + 1
        if not usable:
            if self.tracking_gap_start is None:
                self.tracking_gap_start = now
            self.tracking_gap_frames += 1
        if self.tracking_gap_start is not None:
            expired = (self.tracking_gap_frames > 2
                       or now - self.tracking_gap_start >= .55)
            if expired:
                self.frames.clear()
                self.probabilities.clear()
                self.stable_label = None
                self.stable_count = 0
            if self.no_hands >= 5:
                self.gesture_confirmed = False
                self.last_confirmed = None
            if usable:
                self.tracking_gap_start = None
                self.tracking_gap_frames = 0
            elif not expired and self.frames:
                return 'tracking-gap'
        return 'observing' if usable else 'quality-rejected'

    # Libera los detectores de video e imagen independientes de esta sesión.
    def close(self):
        for detector in self.detail_detectors.values():
            detector.close()
        if self.inverted_detector is not None:
            self.inverted_detector.close()
        if self.hand_detector is not None:
            self.hand_detector.close()
        if self.snapshot_detector is not None:
            self.snapshot_detector.close()
        if self.holistic is not None:
            self.holistic.close()


# Coordina cámaras, modelos, salas temporales, administración y entrenamiento del servidor.
class Runtime:
    # Crea el bloqueo compartido, las cachés y los registros en memoria del servidor.
    def __init__(self):
        self.lock = threading.RLock()
        self.startup_models_ready = threading.Event()
        self.startup_models_ready.set()
        self.sessions = {}
        self.conversations = {}
        self.models = {}
        self.model_signatures = {}
        self.snapshot_models = {}
        self.process = None
        self.job_id = None
        self.recovered_training_checked = False
        self.recovered_training_pid = None
        self.admin = AdminGuard(WEB_DATA / 'admin' / 'auth.json', default_config=TEAM_CONFIG)

    # Abre una sala temporal con llaves separadas y genera su QR local cuando existe origen móvil HTTPS.
    def create_conversation(self, language, phone_origin):
        language_info(language)
        now = time.monotonic()
        for key, room in list(self.conversations.items()):
            if now >= room['expires']:
                self.close_conversation(key)
        if len(self.conversations) >= 8:
            oldest = min(self.conversations, key=lambda key: self.conversations[key]['created'])
            self.close_conversation(oldest)
        identifier = secrets.token_hex(12)
        desktop_key = secrets.token_urlsafe(32)
        phone_key = secrets.token_urlsafe(32)
        room = {'id': identifier, 'language': language, 'desktop_key': desktop_key,
                'phone_key': phone_key, 'created': now, 'expires': now + 3600,
                'phone_session': None, 'messages': [], 'next_message': 1, 'sign_drafts': {}}
        self.conversations[identifier] = room
        phone_url = f"{phone_origin}/phone/{identifier}/{phone_key}" if phone_origin else None
        qr = None
        if phone_url:
            import cv2
            matrix = cv2.QRCodeEncoder_create().encode(phone_url)
            matrix = cv2.copyMakeBorder(matrix, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
            matrix = cv2.resize(matrix, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            ok, encoded = cv2.imencode('.png', matrix)
            if not ok:
                self.conversations.pop(identifier, None)
                raise ValueError('No se pudo generar el código QR.')
            qr = base64.b64encode(encoded.tobytes()).decode('ascii')
        return {'room': identifier, 'desktop_key': desktop_key, 'phone_url': phone_url,
                'qr_png': qr, 'expires_in': 3600, 'messages': [], 'phone_connected': False}

    # Valida la llave del participante y elimina salas vencidas antes de permitir su uso.
    def conversation(self, identifier, key, side):
        room = self.conversations.get(str(identifier or ''))
        key_name = 'desktop_key' if side == 'desktop' else 'phone_key'
        if room and time.monotonic() >= room['expires']:
            self.close_conversation(room['id'])
            room = None
        if (not room or not isinstance(key, str)
                or not secrets.compare_digest(room[key_name], key)):
            raise ValueError('El enlace de conversación venció o dejó de ser válido. Crea uno nuevo.')
        return room

    # Normaliza y añade un mensaje al historial acotado en memoria de la sala.
    def conversation_message(self, room, sender, text, kind='text', confidence=None):
        message = ' '.join(str(text or '').split())
        if not message or len(message) > 300:
            raise ValueError('Escribe un mensaje de 1 a 300 caracteres.')
        if kind not in ('text', 'sign'):
            raise ValueError('Tipo de mensaje no válido.')
        item = {'id': room['next_message'], 'sender': sender, 'text': message,
                'kind': kind, 'confidence': confidence, 'created_at': time.time()}
        room['next_message'] += 1
        room['messages'].append(item)
        room['messages'] = room['messages'][-80:]
        return item

    # Publica cada borrador que lleva cuatro segundos sin una seña confirmada nueva.
    def flush_sign_drafts(self, room):
        now = time.monotonic()
        for side, draft in list(room['sign_drafts'].items()):
            if now - draft['updated'] >= 4:
                self.conversation_message(room, draft['sender'], draft['text'], 'sign',
                                          sum(draft['confidence']) / len(draft['confidence']) if draft['confidence'] else None)
                del room['sign_drafts'][side]

    # Acumula letras o palabras por participante y reinicia la pausa de envío automático.
    def append_conversation_sign(self, room, side, text, confidence):
        self.flush_sign_drafts(room)
        text = ' '.join(str(text or '').split())
        if not text or len(text) > 300:
            raise ValueError('La seña reconocida no tiene un texto válido.')
        drafts = room['sign_drafts']
        draft = drafts.get(side)
        separator = '' if draft and draft['last_letter'] and len(text) == 1 else ' '
        combined = draft['text'] + separator + text if draft else text
        if len(combined) > 300:
            self.conversation_message(room, draft['sender'], draft['text'], 'sign',
                                      sum(draft['confidence']) / len(draft['confidence']) if draft['confidence'] else None)
            draft = None
            combined = text
        new_score = [] if confidence is None else [float(confidence)]
        scores = draft['confidence'] + new_score if draft else new_score
        drafts[side] = {'text': combined, 'updated': time.monotonic(), 'confidence': scores,
                        'last_letter': len(text) == 1, 'sender': 'Señas' if side == 'phone' else 'Computadora'}

    # Publica borradores vencidos y devuelve mensajes, conexión y tiempo restante de una sala.
    def conversation_state(self, room):
        self.flush_sign_drafts(room)
        return {'messages': room['messages'][-80:], 'phone_connected': room['phone_session'] in self.sessions,
                'expires_in': max(0, int(room['expires'] - time.monotonic())),
                'language': room['language'], 'locale': language_info(room['language'])['locale'],
                'sign_drafts': {side: draft['text'] for side, draft in room['sign_drafts'].items()},
                'sign_pause_seconds': 4}

    # Revoca la sala y libera la cámara del teléfono asociada.
    def close_conversation(self, identifier):
        room = self.conversations.pop(identifier, None)
        if room and room.get('phone_session'):
            item = self.sessions.pop(room['phone_session'], None)
            if item:
                item.close()

    # Atiende mensajes, cámara y fotos del teléfono limitados a la sala autorizada por su llave.
    def phone_request(self, route, payload):
        room = self.conversation(payload.get('room'), payload.get('key'), 'phone')
        if route == '/api/phone/state':
            return self.conversation_state(room)
        if route == '/api/phone/message':
            self.conversation_message(room, 'Teléfono', payload.get('text'))
            return self.conversation_state(room)
        if route == '/api/phone/join':
            if self.training():
                raise ValueError('Espera a que termine el entrenamiento para probar señas.')
            sid = room.get('phone_session')
            if sid in self.sessions:
                return {'joined': True, 'session': sid, **self.conversation_state(room)}
            if not self.load_model(room['language']):
                raise ValueError('Este perfil aún no tiene un modelo entrenado para reconocer señas.')
            sid = self.new_session(room['language'], 'Teléfono', 'móvil', quality_gate=True)
            room['phone_session'] = sid
            self.session(sid).owner_id = 'phone:' + room['id']
            return {'joined': True, 'session': sid, **self.conversation_state(room)}
        if route == '/api/phone/snapshot':
            sid = room.get('phone_session')
            if sid not in self.sessions:
                raise ValueError('Activa la cámara del teléfono antes de tomar la foto.')
            result = self.analyze_snapshot(sid, payload.get('image'))
            if result.get('accepted'):
                self.append_conversation_sign(room, 'phone', result['label']['text'], None)
            result['conversation'] = self.conversation_state(room)
            return result
        if route == '/api/phone/frame':
            sid = room.get('phone_session')
            if sid not in self.sessions:
                raise ValueError('Activa la cámara del teléfono para empezar.')
            result = self.frame({'session': sid, 'image': payload.get('image'), 'mode': 'preview' if payload.get('snapshot_mode') is True else 'test'})
            if result.get('confirmed'):
                self.append_conversation_sign(room, 'phone', result['confirmed'].get('text'), result.get('confidence'))
            result['conversation'] = self.conversation_state(room)
            return result
        if route == '/api/phone/leave':
            sid = room.get('phone_session')
            if sid:
                item = self.sessions.pop(sid, None)
                if item:
                    item.close()
                room['phone_session'] = None
            return {'left': True, **self.conversation_state(room)}
        raise ValueError('Acción de teléfono desconocida.')

    # Desactiva un modelo que contiene una clase modificada para evitar reconocer datos retirados.
    def invalidate_label_model(self, language, label):
        manifest = model_dir(language) / 'current.json'
        if not manifest.is_file():
            return False
        selection = json.loads(manifest.read_text(encoding='utf-8'))
        version = selection['version']
        if not isinstance(version, str) or len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
            raise ValueError('La versión del modelo es inválida.')
        metadata = json.loads((manifest.parent / version / 'metadata.json').read_text(encoding='utf-8'))
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

    # Verifica administración y confirmación antes de archivar los ejemplos de una etiqueta.
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

    # Recupera sesiones inactivas y abre una cámara con perfil y procedencia independientes.
    def new_session(self, language, participant="", camera="", quality_gate=False):
        language_info(language)
        now = time.monotonic()
        for key, item in list(self.sessions.items()):
            if now - item.touched >= 15:
                item.close()
                del self.sessions[key]
        if len(self.sessions) >= 4:
            raise ValueError('Ya hay cuatro sesiones abiertas. Detén una cámara y vuelve a intentarlo.')
        identifier = uuid.uuid4().hex
        self.sessions[identifier] = Session(language)
        self.sessions[identifier].quality_gate = quality_gate is True
        self.sessions[identifier].provenance = {'session_id': identifier, 'participant': str(participant or '')[:80], 'camera': str(camera or '')[:80]}
        return identifier

    # Valida una sesión de cámara y renueva su marca de actividad.
    def session(self, identifier):
        if identifier not in self.sessions:
            raise ValueError('La sesión terminó. Detén y vuelve a activar la cámara.')
        item = self.sessions[identifier]
        item.touched = time.monotonic()
        return item

    # Comprueba límites y operaciones pendientes y prepara una secuencia nueva para grabar.
    def start_capture(self, identifier, consecutive=False, batch_id=""):
        if not isinstance(consecutive, bool):
            raise ValueError('La opción de grabación consecutiva no es válida.')
        item = self.session(identifier)
        pending = pending_drafts(item.language)
        if item.owner_id:
            pending = [d for d in pending if team_access.draft_owner(item.language, d['id']) == item.owner_id]
        if pending and not consecutive:
            raise ValueError('Pon nombre o descarta la grabación pendiente antes de hacer otra, o usa una tanda consecutiva.')
        if len(pending) >= 100:
            raise ValueError('Hay 100 tomas pendientes en esta lengua. Nombra y guarda algunas antes de continuar.')
        if self.training() or item.capture:
            raise ValueError('Termina la operación actual antes de grabar otra secuencia.')
        item.reset()
        item.capture = {'points': [], 'hands': 0, 'quality_bad': 0, 'wait_frames': 0}
        item.provenance['batch_id'] = str(batch_id or '')[:80]
        return {'capturing': True}

    # Lee el progreso del entrenamiento y detecta trabajos cuyo proceso se interrumpió.
    def job(self):
        if not self.job_id:
            previous = sorted((WEB_DATA / 'jobs').glob('*.json'), key=lambda p:p.stat().st_mtime, reverse=True)
            if not previous:
                return None
            self.job_id = previous[0].stem
        path = WEB_DATA / 'jobs' / f'{self.job_id}.json'
        state = {'id': self.job_id, 'state': 'running', 'message': 'Preparando entrenamiento…', 'epoch': 0}
        if path.exists():
            state = json.loads(path.read_text(encoding='utf-8'))
        if state.get('state') == 'running' and self.process is None:
            pid = state.get('pid')
            if not process_alive(pid):
                state.update(state='failed', message='El entrenamiento se interrumpió. Los modelos anteriores siguen guardados.')
        if self.process is not None and self.process.poll() is not None and state['state'] == 'running':
            state.update(state='failed', message='El proceso terminó antes de completar el modelo. Revisa web_data/jobs/' + self.job_id + '.log')
        if 'result' in state:
            state['result'] = {k: v for k, v in state['result'].items() if k not in ('posture_vote', 'hand_guard', 'temporal_guard', 'split_hashes', 'split_groups')}
        return state

    # Comprueba si sigue vivo un entrenamiento iniciado o recuperado después de reiniciar.
    def training(self):
        if self.process is not None:
            return self.process.poll() is None
        if not self.recovered_training_checked:
            self.recovered_training_checked = True
            status = self.job() or {}
            if status.get('state') == 'running':
                self.recovered_training_pid = status.get('pid')
        if isinstance(self.recovered_training_pid, int):
            if process_alive(self.recovered_training_pid):
                return True
            self.recovered_training_pid = None
        return False

    # Reúne perfiles, vocabularios, ejemplos, modelos, actividad y trabajo para la interfaz.
    def state(self, *, include_studio=True):
        profiles = []
        for info in LANGUAGES:
            rows = (inventory if include_studio else stored_sample_counts)(info['id'])
            metadata = None
            error = None
            try:
                result = model_metadata(info['id'], public=not include_studio)
                metadata = {k: v for k, v in result[1].items() if k not in ('posture_vote', 'hand_guard', 'temporal_guard', 'split_hashes', 'split_groups')} if result else None
                if not result:
                    manifest = model_dir(info['id']) / 'current.json'
                    if manifest.is_file():
                        selection = json.loads(manifest.read_text(encoding='utf-8'))
                        if selection.get('requires_retraining'):
                            error = selection.get('reason', 'Reentrena el modelo tras borrar ejemplos.')
            except (ValueError, KeyError, OSError) as exc:
                error = str(exc)
            profiles.append({**info, 'labels': rows, 'model': metadata, 'model_error': error, 'drafts': pending_drafts(info['id']) if include_studio else [], 'alphabet': alphabet_status(info['id'], rows, metadata) if include_studio else None})
        old_model = ROOT / 'models' / 'lsm_model.h5'
        return {'profiles': profiles, 'minimum_samples': MIN_SAMPLES, 'recommended_samples': RECOMMENDED_SAMPLES, 'sequence_length': SEQUENCE_LENGTH,
                'target_fps': CAPTURE_FPS, 'token': TOKEN, 'job': self.job() if include_studio else None,
                'legacy_model': old_model.is_file(), 'activity': recent_activity() if include_studio else [], 'api_version': 2}

    # Prepara el comparador de fotos con referencias entrenadas y grabaciones únicas de validación congelada.
    def prepare_snapshot(self, language):
        from snapshot_recognition import build_snapshot
        from web_dataset import load_sequence
        from sign_languages import dataset_dir
        bundle = model_metadata(language)
        if not bundle:
            raise ValueError('Primero necesitas un modelo activo para esta lengua.')
        directory, metadata = bundle
        signature = self.model_signature(language, directory)
        cached = self.snapshot_models.get(language)
        if cached and cached[0] == signature:
            return cached[1]
        split = metadata.get('split_hashes', {})
        reserved = set(split.get('validation', [])) - set(split.get('train', [])) - set(split.get('test', []))
        frames, targets, seen = [], [], set()
        for i, label in enumerate(metadata['labels']):
            for path in sorted((dataset_dir(language) / label['id']).glob('*.npz')):
                try:
                    points = load_sequence(path)
                except (ValueError, KeyError, OSError):
                    continue
                fingerprint = hashlib.sha256(points.tobytes()).hexdigest()
                if fingerprint in reserved and fingerprint not in seen:
                    seen.add(fingerprint)
                    # One real central observation per recording. Frames from
                    # one video are not counted as independent validations.
                    frame = points[len(points)//2]
                    if landmark_quality(frame)['usable']:
                        frames.append(frame); targets.append(i)
        result = build_snapshot(metadata, frames, targets)
        self.snapshot_models[language] = (signature, result)
        return result

    # Crea bajo demanda el detector independiente usado por fotos y recuperación del seguimiento.
    def image_detector(self, item):
        if item.snapshot_detector is None:
            import mediapipe as mp
            item.snapshot_detector = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1,
                                                                   min_detection_confidence=.5)
        return item.snapshot_detector

    def refine_downward_hand(self, item, image, result):
        """Refina dedos en un recorte local, conservando orientación y lateralidad."""
        import cv2
        from types import SimpleNamespace
        from utils import extract_keypoints
        height, width = image.shape[:2]
        now = time.monotonic()
        for side in ('left_hand_landmarks', 'right_hand_landmarks'):
            hand = getattr(result, side, None)
            previous = item.detail_regions.get(side)
            if previous and (now-previous['time'] > .35 or previous['image_size'] != (width,height)):
                item.detail_regions.pop(side,None)
                previous = None
            if hand is None:
                if not previous:
                    continue
                hand = previous['seed']
            xy = np.array([(p.x*width,p.y*height) for p in hand.landmark])
            axis = xy[9] - xy[0]
            inverted = bool(axis[1] >= max(8, .4 * np.linalg.norm(axis)))
            low, high = xy.min(axis=0), xy.max(axis=0)
            size = min(height, width, int(np.ceil(1.9 * max(high-low))))
            if size < 32:
                continue
            center = (low+high)/2
            if previous:
                # Seguir la muñeca actual sin arrastrar las puntas antiguas.
                predicted_center = previous['center'] + xy[0] - previous['wrist']
                center = .75 * predicted_center + .25 * center
                size = int(round(.8 * previous['size'] + .2 * size))
            size = min(size,height,width)
            x,y = np.clip((center-size/2).astype(int), 0, [width-size,height-size])
            # Si el recorte suavizado dejaría dedos fuera, usar el encuadre actual.
            if np.any(low < [x,y]) or np.any(high > [x+size,y+size]):
                size = min(height,width,max(32,int(np.ceil(1.9*max(high-low)))))
                center = (low+high)/2
                x,y = np.clip((center-size/2).astype(int),0,[width-size,height-size])
            # Un recorte fijo permite que Hands siga la misma palma entre
            # cuadros. No alternar dos manos dentro de un único tracker.
            from capture_quality import stable_hand_crop
            crop = stable_hand_crop(xy, (int(x), int(y), int(size)),
                                    item.detail_crops.get((side, inverted)), (width, height), now)
            x, y, size = crop[:3]
            old_crop = item.detail_crops.get((side, inverted))
            detector_key = (side, inverted)
            detector = item.detail_detectors.get(detector_key)
            if detector is None:
                import mediapipe as mp
                detector = mp.solutions.hands.Hands(static_image_mode=False,
                    max_num_hands=1, model_complexity=1)
                item.detail_detectors[detector_key] = detector
            elif old_crop is None or crop[:3] != old_crop[:3] or now-old_crop[3] > .35:
                # Cambió el sistema local de coordenadas: descartar el ROI
                # interno del tracker antes de procesar la imagen actual.
                detector.reset()
            item.detail_crops[detector_key] = crop
            roi = image[y:y+size,x:x+size]
            rgb = cv2.cvtColor(cv2.rotate(roi,cv2.ROTATE_180) if inverted else roi,cv2.COLOR_BGR2RGB)
            found = detector.process(rgb).multi_hand_landmarks
            if not found:
                continue
            candidate = found[0]
            for point in candidate.landmark:
                point.x = (x+(1-point.x if inverted else point.x)*size)/width
                point.y = (y+(1-point.y if inverted else point.y)*size)/height
                point.z *= size/width
            wrist = candidate.landmark[0]
            if np.linalg.norm([wrist.x*width-xy[0,0],wrist.y*height-xy[0,1]]) > .18*size:
                continue
            # Comparar solo detecciones del cuadro actual, nunca contra una
            # postura antigua: la mano puede moverse libremente entre cuadros.
            if getattr(result, side, None) is not None:
                from capture_quality import consistent_palm
                candidate_xy = np.array([(p.x*width, p.y*height) for p in candidate.landmark])
                if not consistent_palm(xy, candidate_xy):
                    continue
            fields = {k:getattr(result,k) for k in ('pose_landmarks','face_landmarks','left_hand_landmarks','right_hand_landmarks')}
            fields[side] = candidate
            refined = SimpleNamespace(**fields)
            if landmark_quality(extract_keypoints(refined))['usable']:
                # Guardar solo el encuadre de una detección actual. Nunca devolver
                # puntos viejos cuando la nueva imagen no contiene una mano.
                item.detail_regions[side] = dict(time=now,center=np.array([x+size/2,y+size/2]),
                    wrist=np.array([wrist.x*width,wrist.y*height]),size=size,
                    seed=candidate,image_size=(width,height))
                result = refined
        return result

    # Reintenta una pérdida reciente sobre la misma imagen, con frecuencia acotada y sin inventar puntos.
    def recover_hand_tracking(self, item, mirrored, result, now):
        from utils import hands_detected, mediapipe_detection, extract_keypoints
        if hands_detected(result) and landmark_quality(extract_keypoints(result))['usable']:
            item.last_hand_seen = now
            refined = self.refine_downward_hand(item, mirrored, result)
            return refined, refined is not result
        if item.detail_regions:
            recovered = self.refine_downward_hand(item, mirrored, result)
            if recovered is not result:
                item.last_hand_seen = now
                return recovered, True
        # Durante un giro Holistic puede perder la mano aunque siga viendo
        # la muñeca. Hands busca la palma directamente en el fotograma actual.
        pose = getattr(result, 'pose_landmarks', None)
        if hasattr(result, 'left_hand_landmarks'):
            if now < item.hand_recovery_after:
                return result, False
            import cv2
            import mediapipe as mp
            from types import SimpleNamespace
            rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            height, width = mirrored.shape[:2]
            slots = {}
            # Cada detector mantiene su propio seguimiento en una orientación fija.
            # Si el giro funcionó, probarlo primero en los siguientes fotogramas.
            prefer_inverted = now < item.inverted_until
            for inverted in (prefer_inverted, not prefer_inverted):
                attr = 'inverted_detector' if inverted else 'hand_detector'
                detector = getattr(item, attr)
                if detector is None:
                    detector = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1)
                    setattr(item, attr, detector)
                detection = detector.process(cv2.rotate(rgb, cv2.ROTATE_180) if inverted else rgb)
                labels = getattr(detection, 'multi_handedness', None) or ()
                for index, hand in enumerate(detection.multi_hand_landmarks or ()):
                    if inverted:
                        # Deshacer la rotación en x/y; z está referido a la muñeca.
                        for point in hand.landmark:
                            point.x, point.y = 1. - point.x, 1. - point.y
                    wrist = hand.landmark[0]
                    distances = sorted((np.hypot(wrist.x - pose.landmark[i].x,
                                                  (wrist.y - pose.landmark[i].y) * height / width), side)
                                       for i, side in ((15, 'left'), (16, 'right'))
                                       if pose and pose.landmark[i].visibility >= .5)
                    if distances:
                        distance, side = distances[0]
                        if len(distances) > 1 and distances[1][0] - distance < .025:
                            continue
                    if not distances or distance > .15:
                        if index >= len(labels) or not labels[index].classification:
                            continue
                        label = labels[index].classification[0]
                        side = label.label.lower()
                        if label.score < .9 or side not in ('left', 'right'):
                            continue
                        if any(visible_side == side for _, visible_side in distances):
                            continue
                        distance = 1 - label.score
                    if side not in slots or distance < slots[side][0]:
                        slots[side] = (distance, hand)
                if slots:
                    item.inverted_until = now + .8 if inverted else 0.
                    break
            if slots:
                recovered = SimpleNamespace(pose_landmarks=pose, face_landmarks=result.face_landmarks,
                    left_hand_landmarks=slots['left'][1] if 'left' in slots else result.left_hand_landmarks,
                    right_hand_landmarks=slots['right'][1] if 'right' in slots else result.right_hand_landmarks)
                if landmark_quality(extract_keypoints(recovered))['usable']:
                    item.last_hand_seen = now
                    return self.refine_downward_hand(item, mirrored, recovered), True
            # Reintentar en el próximo cuadro durante una toma o pérdida reciente.
            item.hand_recovery_after = now if item.capture or now-item.last_hand_seen < .8 else now + .10
        if not item.last_hand_seen or now-item.last_hand_seen > .8 or now < item.pose_recovery_after:
            return result, False
        item.pose_recovery_after = now + 1.
        _, recovered = mediapipe_detection(mirrored, self.image_detector(item))
        if not hands_detected(recovered) or not landmark_quality(extract_keypoints(recovered))['usable']:
            return result, False
        item.last_hand_seen = now
        # A successful independent re-detection can reinitialize a stuck tracker.
        if item.holistic is not None:
            item.holistic.close()
            item.holistic = None
        return recovered, True

    # Valida una imagen, detecta su postura de forma independiente y ejecuta la comparación experimental.
    def analyze_snapshot(self, identifier, encoded):
        import cv2
        from snapshot_recognition import classify_snapshot
        from utils import mediapipe_detection, extract_keypoints
        item = self.session(identifier)
        if self.training() or item.capture:
            raise ValueError('Termina la captura o el entrenamiento antes de comparar una foto.')
        if not isinstance(encoded, str) or len(encoded) > 1500000:
            raise ValueError('Imagen demasiado grande.')
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('Imagen inválida.') from exc
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[0] > 1080 or image.shape[1] > 1920:
            raise ValueError('Imagen inválida o demasiado grande.')
        mirrored = cv2.flip(image, 1)
        _, result = mediapipe_detection(mirrored, self.image_detector(item))
        points = extract_keypoints(result)
        quality = landmark_quality(points)
        warnings = hand_image_quality(mirrored, points)['warnings']
        if not quality['usable'] or warnings:
            return {'accepted':False,'label':None,'alternatives':[], 'reason':warnings[0] if warnings else quality['message']}
        return classify_snapshot(points, self.prepare_snapshot(item.language))

    # Obtiene firmas de manifiesto, metadatos y pesos para invalidar la caché cuando cambian.
    def model_signature(self, language, directory):
        paths = (model_dir(language) / 'current.json', directory / 'metadata.json', directory / 'model.keras')
        return tuple((st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) for st in (path.stat() for path in paths))

    # Reutiliza un modelo sin cambios o valida y carga pesos y filtros compatibles desde disco.
    def load_model(self, language):
        cached = self.models.get(language)
        if cached and language in self.model_signatures:
            try:
                if self.model_signature(language, cached[0]) == self.model_signatures[language]:
                    return cached[1:]
            except OSError:
                pass  # A removed/replaced artifact must pass full validation.
        bundle = model_metadata(language)
        if not bundle:
            return None
        directory, meta = bundle
        cached = self.models.get(language)
        if not cached or cached[0] != directory or language in self.model_signatures:
            import tensorflow as tf
            model = load_inference_model(directory / 'model.keras')
            if tuple(model.input_shape[1:]) != (SEQUENCE_LENGTH, feature_dimension(meta.get('preprocessing'))) or model.output_shape[-1] != len(meta['labels']):
                raise ValueError('Las dimensiones del modelo no coinciden con sus metadatos.')
            # Reutilizar el grafo de inferencia evita ejecutar cada paso LSTM
            # desde Python en cada fotograma. El objeto conserva su API Keras.
            model.call = tf.function(model.call, reduce_retracing=True)
            for guard_name in ('hand_guard', 'temporal_guard'):
                for item in (meta.get(guard_name) or {}).get('classes', []):
                    item['references'] = np.asarray(item['references'], dtype=np.float32)
            if meta.get('posture_vote'):
                meta['posture_vote']['weights'] = np.asarray(meta['posture_vote']['weights'],np.float32)
            self.models[language] = directory, model, meta
        self.model_signatures[language] = self.model_signature(language, directory)
        return self.models[language][1:]

    def retry_longer_motion(self, history, model, metadata, strict):
        """Recover a truncated gesture only when two longer views agree."""
        from motion_features import longer_motion_windows
        moving = set((metadata.get('motion_profile') or {}).get('moving_classes', []))
        if not moving:
            return None
        windows = longer_motion_windows(history, SEQUENCE_LENGTH)
        if not windows:
            return None
        prepared = np.stack([prepare_sequence(w, metadata.get('preprocessing')) for w in windows])
        probabilities = model(prepared, training=False).numpy()
        probabilities = temperature_probabilities(probabilities, metadata.get('temperature', 1.))
        probabilities = apply_vote(prepared, probabilities, metadata.get('posture_vote'))
        winners = probabilities.argmax(axis=1)
        if winners[0] not in moving or not np.all(winners == winners[0]):
            return None
        verdicts = [decision_details(w, p, metadata, strict=strict, test_mode=True)
                    for w,p in zip(windows, probabilities)]
        if not all(v['accepted'] for v in verdicts):
            return None
        # Return the weaker accepted view, never a synthetic averaged score.
        index = int(np.argmin(probabilities[:, winners[0]]))
        return probabilities[index], verdicts[index]

    def retry_viewpoint(self, points, probabilities, model, metadata, strict):
        """Reevalúa una ventana rechazada bajo dos variaciones leves de cámara."""
        original_index = int(np.argmax(probabilities))
        policy = (metadata.get('decision_policy') or {}).get('classes', {})
        if policy.get(str(original_index), {}).get('supported') is False:
            # A repeated inference cannot change calibration support. Avoid
            # two extra model evaluations on every frame of a blocked class.
            return None
        views, _ = augment_viewpoints(points[None], np.array([0]))
        prepared = np.stack([prepare_sequence(view, metadata.get('preprocessing')) for view in views[1:]])
        variants = model(prepared, training=False).numpy()
        variants = temperature_probabilities(variants, metadata.get('temperature', 1.))
        variants = apply_vote(prepared, variants, metadata.get('posture_vote'))
        consensus = (probabilities + variants.sum(axis=0)) / 3
        original_index = int(np.argmax(probabilities))
        if consensus.argmax() != original_index or not np.all(variants.argmax(axis=1) == original_index):
            return None
        verdict = decision_details(points, consensus, metadata, strict=strict, test_mode=True)
        return (consensus, verdict) if verdict['accepted'] else None

    def retry_i_with_other_hand(self, points, model, metadata, strict):
        """Verifica I en la otra mano solo después de un rechazo normal."""
        from motion_features import motion_summary
        from recognition_features import regions, hand_descriptor, nearest_distance

        labels = metadata['labels']
        index = next((i for i, label in enumerate(labels) if label['text'].casefold() == 'i'), None)
        guard = metadata.get('hand_guard') or {}
        if index is None or guard.get('version') != 'finger-contrast-v2':
            return None
        left, right = regions(points)[-2:]
        if np.any(left) or np.mean(np.any(right, axis=(1,2))) < .9:
            return None
        if motion_summary(points)['extent'] > .12:
            return None

        reflected = points.copy()
        pose, face, left, right = regions(reflected)
        for region in (pose, face, left, right):
            region[...,0] = np.where(np.any(region[...,:3] != 0,axis=-1),1-region[...,0],0)
        left[:], right[:] = right.copy(), left.copy()
        for a,b in zip(range(11,33,2),range(12,33,2)):
            pose[:,[a,b]] = pose[:,[b,a]]

        descriptor = hand_descriptor(reflected, True, guard.get('descriptor_version'))
        distances = [nearest_distance(descriptor,row['references']) for row in guard['classes']]
        own = distances[index]
        competitors = [distance for j,distance in enumerate(distances) if j != index]
        if not competitors or own > .7 * guard['classes'][index]['max_distance'] or own > .8 * min(competitors):
            return None

        prepared = prepare_sequence(reflected,metadata.get('preprocessing'))
        probabilities = model(prepared[None,...],training=False).numpy()[0]
        probabilities = temperature_probabilities(probabilities,metadata.get('temperature',1.))
        probabilities = apply_vote(prepared,probabilities,metadata.get('posture_vote'))
        if probabilities.argmax() != index or probabilities[index] < (.95 if strict else .90):
            return None
        verdict = decision_details(reflected,probabilities,metadata,strict=strict,test_mode=True)
        return (probabilities,verdict) if verdict['accepted'] else None

    # Procesa un fotograma: seguimiento, calidad, captura o inferencia con filtros y confirmación temporal.
    def frame(self, payload):
        if 'confirmation_ack' not in payload:
            return self._frame(payload)
        session = self.session(payload.get('session'))
        pending = session.pending_confirmation
        if pending and payload.get('confirmation_ack') == pending['confirmation_id']:
            session.pending_confirmation = None
        result = self._frame(payload)
        if result.get('confirmed'):
            session.pending_confirmation = {
                'confirmation_id': uuid.uuid4().hex,
                'confirmed': result['confirmed'],
                'confirmed_confidence': result['confidence'],
            }
        if session.pending_confirmation:
            result.update(session.pending_confirmation)
        return result

    def _frame(self, payload):
        # El intervalo de transporte excluye TODO el trabajo local: inicializar
        # MediaPipe, cargar el modelo, inferir y escribir la toma en disco.
        session = self.session(payload.get('session'))
        try:
            return self._process_frame(payload)
        finally:
            session.last_frame = time.monotonic()

    def _process_frame(self, payload):
        now = time.monotonic()
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
        gap = now - session.last_frame if session.last_frame else 0
        interrupted = gap > 1.0 and session.capture is not None
        if gap > 1.0:
            session.reset()
            session.capture = None
        if session.holistic is None:
            import mediapipe as mp
            session.holistic = mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=.5, min_tracking_confidence=.5)
        mirrored = cv2.flip(image, 1)
        _, results = mediapipe_detection(mirrored, session.holistic)
        results, recovered = self.recover_hand_tracking(session, mirrored, results, now)
        points = extract_keypoints(results)
        hands = hands_detected(results)
        output = {'hands': hands, 'tracking_recovered': recovered, 'buffer': min(len(session.frames), SEQUENCE_LENGTH), 'captured': 0, 'draft_id': None,
                  'candidate': None, 'confirmed': None, 'confidence': 0, 'stability': 0,
                  'stability_required': STABILITY_FRAMES, 'rejection_reason': None, 'rejection_code': None}
        if payload.get('show_hand_rig') is True:
            output['hand_rig'] = [
                [[round(float(p.x), 5), round(float(p.y), 5)] for p in hand.landmark]
                for hand in (results.left_hand_landmarks, results.right_hand_landmarks)
                if hand is not None
            ]
        quality = landmark_quality(points)
        visual = hand_image_quality(mirrored, points)
        quality['warnings'] = visual['warnings']
        quality['image_scope'] = visual['scope']
        output.update(quality=quality, phase='waiting', alternatives=[], latency_ms=0)
        if interrupted:
            # La toma parcial no se guarda ni se reinicia a espaldas del usuario.
            # Mantener el visor permite repetirla sin volver a abrir la cámara.
            output.update(phase='capture-interrupted', capture_error=
                'La toma se interrumpió entre fotogramas y no se guardó. La cámara sigue activa; vuelve a grabar la seña completa.')
            return output
        if session.quality_gate and visual['warnings']:
            quality['usable'] = False
            quality['message'] = visual['warnings'][0]
        if session.capture is not None:
            capture = session.capture
            if session.quality_gate and not capture['points'] and not quality['usable']:
                capture['wait_frames'] += 1
                output['phase'] = 'waiting-for-hands'
                if capture['wait_frames'] >= 50:
                    session.capture = None
                    raise ValueError('No se pudo iniciar una toma válida. Revisa la luz y el encuadre y vuelve a grabar.')
                return output
            capture['quality_bad'] = capture.get('quality_bad', 0) + int(not quality['usable'])
            output['phase'] = 'capturing'
            capture['points'].append(points)
            capture['hands'] += int(hands)
            output['captured'] = len(capture['points'])
            if len(capture['points']) == SEQUENCE_LENGTH:
                session.capture = None
                if capture['hands'] < SEQUENCE_LENGTH * .8:
                    raise ValueError('Las manos no fueron visibles en suficientes fotogramas. Repite la seña.')
                if session.quality_gate and capture['quality_bad'] > SEQUENCE_LENGTH * .2:
                    raise ValueError('La toma perdió calidad o parte de las manos. Repite con buena luz y los dedos dentro del encuadre.')
                if session.quality_gate and sequence_quality(capture['points'])['boundary_warning']:
                    output['capture_warning'] = 'La toma terminó con movimiento. Revisa que la seña esté completa antes de integrarla.'
                output['draft_id'] = save_draft(session.language, capture['points'], session.provenance)
                session.reset()
            return output
        if payload.get('mode') != 'test':
            return output
        usable = hands and (quality['usable'] or not session.quality_gate)
        tracking_phase = session.update_recognition_tracking(hands, usable, now)
        if not usable:
            output['buffer'] = min(len(session.frames), SEQUENCE_LENGTH)
            output['phase'] = tracking_phase
            output['rejection_code'] = 'capture_quality'
            output['rejection_reason'] = quality['message']
            return output
        session.frames.append(points)
        output['buffer'] = min(len(session.frames), SEQUENCE_LENGTH)
        bundle = self.load_model(session.language)
        if not bundle or len(session.frames) < SEQUENCE_LENGTH:
            return output
        model, metadata = bundle
        stability_frames = max(1, int(metadata.get('stability_frames', STABILITY_FRAMES)))
        output['stability_required'] = stability_frames
        model_identity = (metadata.get('version'), metadata.get('created_at'))
        if session.last_model != model_identity:
            session.probabilities.clear()
            session.stable_label = None
            session.stable_count = 0
            session.gesture_confirmed = False
            session.last_confirmed = None
            session.cooldown = 0
            session.last_inference = 0
            session.last_model = model_identity
        if now - session.last_inference < .10:
            output['phase'] = 'observing'
            output['inference_skipped'] = True
            return output
        session.last_inference = now
        raw_sequence = np.stack(session.frames)[-SEQUENCE_LENGTH:]
        sequence = prepare_sequence(raw_sequence, metadata.get('preprocessing'))
        probs = model(sequence[None, ...], training=False).numpy()[0]
        probs = temperature_probabilities(probs, metadata.get('temperature', 1.))
        probs = apply_vote(sequence, probs, metadata.get('posture_vote'))
        if metadata.get('decision_input') == 'single-window-v1':
            # Calibrate and decide on the same window. Consecutive accepted
            # windows below provide stability without mixing prior gestures.
            session.probabilities.clear()
        else:
            # Preserve the decision behavior of previously saved models.
            session.probabilities.append(probs)
            probs = np.mean(session.probabilities, axis=0)
        strict = payload.get('strict') is True
        details = decision_details(raw_sequence, probs, metadata, strict=strict, test_mode=True)
        # Un reintento puede reforzar la lectura original, pero un desacuerdo
        # entre manos/ángulos debe esperar una nueva ventana estable.
        original_index = int(np.argmax(probs))
        original_label = metadata['labels'][original_index]['text'].casefold()
        held_j = session.gesture_confirmed and any(
            label['id'] == session.last_confirmed and label['text'].casefold() == 'j'
            for label in metadata['labels']
        )
        if not details['accepted'] and original_label == 'i' and not held_j:
            retry = self.retry_i_with_other_hand(raw_sequence, model, metadata, strict)
            if retry is not None and int(np.argmax(retry[0])) == original_index:
                probs, details = retry
        if not details['accepted']:
            retry = self.retry_longer_motion(np.stack(session.frames), model, metadata, strict)
            if retry is not None:
                probs, details = retry
                output['temporal_window'] = 'extended'
        if not details['accepted'] and metadata.get('decision_input') == 'single-window-v1':
            retry = self.retry_viewpoint(raw_sequence, probs, model, metadata, strict)
            if retry is not None:
                probs, details = retry
        index = int(np.argmax(probs))
        label = metadata['labels'][index]
        confidence = float(probs[index])
        output.update(candidate=label, confidence=confidence, phase='observing',
                      alternatives=[{**metadata['labels'][int(i)], 'confidence': float(probs[i])} for i in np.argsort(probs)[-3:][::-1]])
        decision = details['accepted']
        output['rejection_reason'] = details.get('reason')
        output['rejection_code'] = details.get('code')
        if not decision:
            # Keep the model proposal visible without treating it as accepted.
            output['proposed_candidate'] = label
            output['candidate'] = None
            if not output['rejection_reason']:
                output['rejection_reason'] = 'Seña ambigua. Mantén el gesto o vuelve a empezar.'
            output['phase'] = 'uncertain'
            session.stable_label, session.stable_count = None, 0
        elif session.stable_label == label['id']:
            session.stable_count += 1
        else:
            session.stable_label, session.stable_count = label['id'], 1
        immediate_test = payload.get('mode') == 'test' and confidence >= .90 - 1e-7 and decision
        switching = session.gesture_confirmed and label['id'] != session.last_confirmed
        # A new sign can follow the previous one without a no-hands gap.
        # Isolated high-confidence frames are not stable gestures. Require
        # three observations even for the first sign; transitions retain the
        # calibrated stability requirement. No confidence threshold is reduced.
        required = max(3, stability_frames) if switching else (3 if immediate_test else max(3, stability_frames))
        output['stability_required'] = required
        output['stability'] = min(session.stable_count, required)
        if decision and session.gesture_confirmed and label['id'] == session.last_confirmed:
            output['phase'] = 'held'
        ready = decision and session.stable_count >= required
        if ready and (switching or (not session.gesture_confirmed and now >= session.cooldown)):
            output['phase'] = 'confirmed'
            session.gesture_confirmed = True
            output['confirmed'] = label
            session.last_confirmed = label['id']
            session.cooldown = now + COOLDOWN_SECONDS
        output['latency_ms'] = round((time.monotonic() - now) * 1000)
        return output

    # Valida clases, épocas y estado y lanza el entrenamiento en un proceso separado.
    def start_training(self, language, epochs, label_ids=None, strategy="baseline"):
        language_info(language)
        if self.training():
            raise ValueError('Ya hay un entrenamiento en curso.')
        if strategy not in ('baseline', 'discriminative', 'contrastive', 'trajectory', 'robust', 'torso'):
            raise ValueError('Estrategia de entrenamiento desconocida.')
        if not isinstance(epochs, int) or not 5 <= epochs <= 200:
            raise ValueError('Selecciona entre 5 y 200 épocas.')
        from web_train import retained_label_ids
        label_ids = retained_label_ids(language, label_ids)
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
                                             '--language', language, '--epochs', str(epochs), '--job-id', self.job_id, '--strategy', strategy, *selection_args],
                                            cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        return {'job_id': self.job_id}


RUNTIME = Runtime()


def check_camera_dependencies():
    """Comprueba que las librerías nativas puedan cargarse antes de abrir una sesión."""
    python = r'.\.venv\Scripts\python.exe' if os.name == 'nt' else '.venv/bin/python'
    for module, name in (('cv2', 'OpenCV'), ('mediapipe', 'MediaPipe')):
        try:
            import_module(module)
        except (ImportError, OSError) as error:
            raise ValueError(
                f'El Python de Seña no pudo cargar {name}: {error}. '
                f'Desde la carpeta LSM, ejecuta {python} -m pip install -r requirements.txt. '
                f'Reinicia con {python} web_app.py.'
            ) from error


# Implementa las rutas HTTP, la entrega de recursos y los controles de origen y autorización.
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    disable_nagle_algorithm = True
    # Configura la conexión y el tiempo de espera antes de recibir peticiones HTTP.
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    # Registra solicitudes evitando ruido por fotogramas y exposición de llaves de salas en rutas.
    def log_message(self, format, *args):
        # Evitar una línea por cada fotograma.
        if not any(route in str(args) for route in ('/api/frame', '/api/phone/', '/phone/')):
            super().log_message(format, *args)

    # Envía JSON o bytes con longitud, cabeceras de seguridad y cierre ante errores HTTP.
    def send_data(self, status, data, content_type='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        if status >= 400:
            self.close_connection = True
            self.send_header('Connection', 'close')
        if getattr(self, 'session_cookie', None):
            self.send_header('Set-Cookie', self.session_cookie)
            self.session_cookie = None
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Permissions-Policy', 'camera=(self), microphone=()')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def reject_post_access(self, route, status, code, message):
        if route == '/api/session':
            self.log_message('Apertura de cámara rechazada: %s', code)
        return self.send_data(status, {'code': code, 'error': message})

    # Permite únicamente los Hosts configurados para la aplicación principal.
    def trusted_host(self):
        host = self.headers.get('Host', '')
        public = getattr(self.server, 'public_origin', None)
        return host == urlsplit(public).netloc if public else host in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    # Comprueba Host y, en mutaciones, Origin exactos del acceso móvil HTTPS.
    def trusted_phone_host(self, origin=False):
        phone_origin = getattr(self.server, 'phone_origin', None)
        if not phone_origin or self.headers.get('Host', '') != urlsplit(phone_origin).netloc:
            return False
        return not origin or self.headers.get('Origin') == phone_origin

    # Entrega la aplicación, la página móvil autorizada, recursos permitidos y estado autenticado.
    def do_GET(self):
        route = urlsplit(self.path).path
        phone_asset = route in {'/static/phone.js', '/static/speech.js', '/static/styles.css', '/static/vendor/manrope.woff2'} and self.trusted_phone_host()
        if not self.trusted_host() and not phone_asset and not (route.startswith('/phone/') and self.trusted_phone_host()):
            return self.send_data(403, {'error': 'Abre la dirección configurada para este servidor de Seña.'})
        if route.startswith('/phone/'):
            parts = route.strip('/').split('/')
            if (len(parts) != 3 or not self.trusted_phone_host()):
                return self.send_data(404, {'error': 'Enlace de teléfono no disponible.'})
            try:
                with RUNTIME.lock:
                    RUNTIME.conversation(parts[1], parts[2], 'phone')
            except ValueError:
                return self.send_data(410, 'Este enlace de conversación venció. Genera otro desde Seña.', 'text/plain; charset=utf-8')
            return self.send_data(200, (ROOT / 'static' / 'phone.html').read_bytes(), 'text/html; charset=utf-8')
        if route == '/api/auth/status':
            return self.send_data(200, team_access.status(self, ACCOUNTS))
        if route.startswith('/api/'):
            auth = team_access.authenticate(self, ACCOUNTS)
            if not auth:
                return self.send_data(401, {'error': 'Se perdió el acceso. Recarga la página e inténtalo de nuevo.'})
        if route == '/api/state':
            with RUNTIME.lock:
                return self.send_data(200, team_access.shared_state(RUNTIME, auth['user'], auth['csrf']))
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/static/speech.js': ('speech.js', 'text/javascript; charset=utf-8'),
                 '/static/accounts.js': ('accounts.js', 'text/javascript; charset=utf-8'),
                 '/static/conversation.js': ('conversation.js', 'text/javascript; charset=utf-8'),
                 '/static/phone.js': ('phone.js', 'text/javascript; charset=utf-8'),
                 '/static/studio.js': ('studio.js', 'text/javascript; charset=utf-8'),
                 '/static/continuous.js': ('continuous.js', 'text/javascript; charset=utf-8'),
                 '/static/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/static/styles.css': ('styles.css', 'text/css; charset=utf-8')}
        if route.startswith('/static/vendor/'):
            name = route.removeprefix('/static/vendor/')
            if name in {'manrope.woff2','lucide.svg','manrope-latin.woff2'}:
                return self.send_data(200, (ROOT / 'static' / 'vendor' / name).read_bytes(), 'font/woff2' if name.endswith('.woff2') else 'image/svg+xml')
        if route not in files:
            return self.send_data(404, {'error': 'Recurso no encontrado.'})
        filename, mime = files[route]
        self.send_data(200, (ROOT / 'static' / filename).read_bytes(), mime)

    # Valida tamaño, JSON, origen, token y permisos antes de ejecutar cada operación de la API.
    def do_POST(self):
        origin = self.headers.get('Origin')
        expected = {getattr(self.server, 'public_origin', None)} if getattr(self.server, 'public_origin', None) else {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        route = urlsplit(self.path).path
        if route.startswith('/api/phone/'):
            if not self.trusted_phone_host(origin=True):
                return self.send_data(403, {'error': 'Origen de teléfono no autorizado.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 2_000_000:
                    raise ValueError('Solicitud vacía o demasiado grande.')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Solicitud no válida.')
                with RUNTIME.lock:
                    result = RUNTIME.phone_request(route, payload)
                return self.send_data(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                return self.send_data(400, {'error': str(exc)})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return self.send_data(500, {'error': f'No se pudo completar la operación: {exc}'})
        auth = team_access.authenticate(self, ACCOUNTS, require_user=not route.startswith('/api/auth/'))
        if not auth:
            return self.reject_post_access(route, 401, 'access_expired',
                                           'Se perdió el acceso. Recarga la página e inténtalo de nuevo.')
        if not self.trusted_host() or (origin and origin not in expected):
            return self.reject_post_access(route, 403, 'unexpected_origin',
                                           'Abre la URL de Seña que aparece en la terminal directamente en tu navegador.')
        if not team_access.csrf_valid(self, auth):
            return self.reject_post_access(route, 403, 'csrf_mismatch',
                                           'El acceso de esta pestaña cambió. Recarga la página e inténtalo de nuevo.')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            max_length = 48 * 1024 * 1024 if urlsplit(self.path).path == '/api/dataset/import' else 2000000
            if length <= 0 or length > max_length:
                raise ValueError('Solicitud vacía o demasiado grande.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Solicitud no válida.')
            route = urlsplit(self.path).path
            if route.startswith('/api/auth/'):
                return self.send_data(200, team_access.auth_post(self, route, payload, ACCOUNTS, RUNTIME, auth))
            if (route not in ('/api/admin/unlock', '/api/admin/lock', '/api/close')
                    and not RUNTIME.startup_models_ready.wait(timeout=10)):
                return self.send_data(503, {'error': 'El modelo aún se está preparando. Inténtalo en unos segundos.'})
            if route == '/api/session' and payload.get('shared_consent') is True:
                check_camera_dependencies()
            with RUNTIME.lock:
                auth = team_access.authenticate(self, ACCOUNTS)
                if not auth:
                    return self.send_data(401, {'error': 'Se perdió el acceso. Recarga la página e inténtalo de nuevo.'})
                if route in ('/api/admin/unlock', '/api/admin/lock'):
                    result = team_access.studio_access_post(self, route, payload, ACCOUNTS, RUNTIME, auth)
                    return self.send_data(200, result)
                team_access.authorize(RUNTIME, auth['user'], route, payload)
                if route == '/api/conversation/create':
                    if payload.get('shared_consent') is not True:
                        raise ValueError('Confirma que los mensajes y las señas se compartirán en esta conversación temporal.')
                    result = RUNTIME.create_conversation(payload.get('language'), getattr(self.server, 'phone_origin', None))
                elif route == '/api/conversation/state':
                    room = RUNTIME.conversation(payload.get('room'), payload.get('key'), 'desktop')
                    result = RUNTIME.conversation_state(room)
                elif route == '/api/conversation/sign':
                    room = RUNTIME.conversation(payload.get('room'), payload.get('key'), 'desktop')
                    RUNTIME.append_conversation_sign(room, 'desktop', payload.get('text'), payload.get('confidence'))
                    result = RUNTIME.conversation_state(room)
                elif route == '/api/conversation/message':
                    room = RUNTIME.conversation(payload.get('room'), payload.get('key'), 'desktop')
                    RUNTIME.conversation_message(room, 'Computadora', payload.get('text'))
                    result = RUNTIME.conversation_state(room)
                elif route == '/api/conversation/close':
                    room = RUNTIME.conversation(payload.get('room'), payload.get('key'), 'desktop')
                    RUNTIME.close_conversation(room['id'])
                    result = {'closed': True}
                elif route == '/api/session':
                    result = {'session': RUNTIME.new_session(payload.get('language'), (str(payload.get('participant', '')).strip()[:80] if auth['user'].get('local') else 'user:' + auth['user']['id']), payload.get('camera'), payload.get('quality_gate'))}
                    item = RUNTIME.session(result['session'])
                    item.owner_id = auth['user']['id']
                    item.provenance['owner_id'] = auth['user']['id']
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
                    result = RUNTIME.start_capture(payload.get('session'), payload.get('consecutive', False), payload.get('batch_id'))
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
                elif route == '/api/snapshot':
                    result = RUNTIME.analyze_snapshot(payload.get('session'), payload.get('image'))
                elif route == '/api/frame':
                    result = RUNTIME.frame(payload)
                elif route == '/api/admin/delete-recordings':
                    result = RUNTIME.delete_recordings(payload)
                elif route.startswith('/api/dataset/'):
                    result = dataset_operation(RUNTIME, route, payload)
                elif route == '/api/models/list':
                    result = {'versions': versions(payload.get('language'))}
                elif route == '/api/train':
                    result = RUNTIME.start_training(payload.get('language'), payload.get('epochs', 40), payload.get('labels'), payload.get('strategy', 'baseline'))
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


def prepare_startup_models(runtime):
    """Deja que la portada cargue mientras se comprueba el modelo más reciente."""
    try:
        if not runtime.training():
            for profile in LANGUAGES:
                selected = activate_latest_version(profile['id'])
                if selected:
                    print(f"Modelo activo de {profile['id']}: {selected}", flush=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"No se pudo preparar el modelo al arrancar: {error}", file=sys.stderr, flush=True)
    finally:
        runtime.startup_models_ready.set()


# Valida los argumentos de arranque, configura el servidor y libera detectores al cerrarlo.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1', help='Escucha interna; usa un proxy HTTPS para acceso remoto.')
    parser.add_argument('--public-url', help='Origen HTTPS del proxy para el acceso público.')
    parser.add_argument('--phone-origin', help='Origen HTTPS para los enlaces temporales del teléfono; puede coincidir con --public-url.')
    parser.add_argument('--quick-tunnel', action='store_true', help='Crea una URL HTTPS temporal con cloudflared para compartir la app sin Tailscale.')
    parser.add_argument('--no-open', action='store_true', help='No abras el navegador automáticamente al iniciar.')
    args = parser.parse_args()
    cache = WEB_DATA / 'cache' / 'matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    if args.quick_tunnel and (args.public_url or args.phone_origin):
        parser.error('--quick-tunnel crea su propio origen; no lo combines con --public-url ni --phone-origin.')
    if args.quick_tunnel and args.host not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('--quick-tunnel solo debe enlazar el servidor a localhost.')
    if args.public_url:
        parsed = urlsplit(args.public_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            parser.error('--public-url debe ser un origen HTTPS sin ruta, credenciales ni parámetros.')
        args.public_url = 'https://' + parsed.netloc
    if args.phone_origin:
        parsed_phone = urlsplit(args.phone_origin)
        if parsed_phone.scheme != 'https' or not parsed_phone.hostname or parsed_phone.username or parsed_phone.password or parsed_phone.path not in ('', '/') or parsed_phone.query or parsed_phone.fragment:
            parser.error('--phone-origin debe ser un origen HTTPS sin ruta, credenciales ni parámetros.')
        args.phone_origin = 'https://' + parsed_phone.netloc
    if args.public_url and args.phone_origin and args.public_url != args.phone_origin:
        parser.error('Si combinas --public-url y --phone-origin, ambos deben usar el mismo origen HTTPS.')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not args.public_url:
        parser.error('Para escuchar fuera de localhost configura --public-url y un proxy HTTPS.')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    tunnel = None
    try:
        if args.quick_tunnel:
            print('Creando enlace HTTPS temporal…', flush=True)
            tunnel, args.public_url = start_quick_tunnel(args.port)
            args.phone_origin = args.public_url
        server.public_origin = args.public_url
        server.phone_origin = args.phone_origin
        server.local_mode = not args.public_url
        server.local_csrf = secrets.token_urlsafe(32)
        address = args.public_url or f'http://127.0.0.1:{args.port}'
        print(f'\nPágina de Seña: {address}', flush=True)
        print('Modo: acceso público con administración protegida.' if args.public_url else 'Modo: solo esta computadora, sin inicio de sesión.', flush=True)
        if args.quick_tunnel:
            print('La URL es temporal y deja de funcionar al cerrar esta ventana. Compártela solo con quien corresponda.', flush=True)
        if not args.no_open:
            threading.Timer(0.6, webbrowser.open, args=(address,)).start()
        print('Ctrl+C para cerrar.\n', flush=True)
        RUNTIME.startup_models_ready.clear()
        threading.Thread(target=prepare_startup_models, args=(RUNTIME,),
                         name='seña-modelos', daemon=True).start()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if tunnel and tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()
        with RUNTIME.lock:
            for item in RUNTIME.sessions.values():
                item.close()
        if RUNTIME.training():
            print('El entrenamiento continúa en su proceso. Consulta web_data/jobs/.')


if __name__ == '__main__':
    main()
