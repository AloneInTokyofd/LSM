"""Controles HTTP del equipo; ninguna ruta de datos es pública."""
import hmac
import json
import time
from http.cookies import SimpleCookie, CookieError
from accounts import COOKIE, digest
from web_dataset import draft_dir, sequence_metadata, validate_identifier

STUDIO_ROUTES = {'/api/train', '/api/alphabet', '/api/capture',
                 '/api/capture/save', '/api/capture/discard', '/api/admin/delete-recordings'}
CAMERA_ROUTES = {'/api/snapshot', '/api/close', '/api/reset', '/api/capture', '/api/frame'}


# Lee de forma segura el token opaco de la cookie de sesión.
def cookie_token(handler):
    try:
        cookie = SimpleCookie(handler.headers.get('Cookie', ''))
        return cookie[COOKIE].value if COOKIE in cookie else None
    except CookieError:
        return None


# Configura la cookie HttpOnly y SameSite con marca Secure en el despliegue HTTPS autenticado.
def set_session(handler, token):
    secure = bool(getattr(handler.server, 'public_origin', None))
    handler.session_cookie = f'{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=43200' + ('; Secure' if secure else '')


# Proporciona una identidad local cuando las cuentas están desactivadas y la conexión es de loopback.
def local_session(handler):
    if (getattr(handler.server, 'local_mode', False)
            and not getattr(handler.server, 'public_origin', None)
            and handler.client_address[0] in ('127.0.0.1', '::1')):
        return {'user': {'id': 'local', 'name': 'Modo local', 'username': 'local',
                         'role': 'collaborator', 'local': True}, 'csrf': handler.server.local_csrf}
    return None


# Devuelve el estado de acceso local o una sesión de cuenta para la pantalla de entrada.
def status(handler, accounts):
    session = authenticate(handler, accounts)
    if not session:
        token, csrf = accounts.issue(previous=cookie_token(handler))
        set_session(handler, token)
        session = guest_session({'user': None, 'csrf': csrf}, token)
    if session['user'].get('local'):
        return dict(session, setup_required=False, shared=False, registration='disabled')
    return {'user': session['user'], 'csrf': session['csrf'], 'setup_required': not accounts.configured(),
            'shared': True, 'registration': 'invitation', 'guest': session['user'].get('guest', False)}


def guest_session(session, token):
    """Give an anonymous browser a stable, least-privilege identity."""
    if session.get('user'):
        return session
    identifier = 'guest:' + digest(token)[:24]
    session['user'] = {'id': identifier, 'name': 'Visitante', 'username': identifier[-8:],
                       'role': 'collaborator', 'active': True, 'guest': True}
    return session


# Cierra los detectores y elimina todas las sesiones de cámara de un usuario.
def close_user_cameras(runtime, identifier):
    with runtime.lock:
        for key, item in list(runtime.sessions.items()):
            if item.owner_id == identifier:
                item.close()
                del runtime.sessions[key]


# Resuelve la identidad local o valida la sesión de cuenta requerida por la ruta.
def authenticate(handler, accounts, require_user=True):
    local = local_session(handler)
    token = cookie_token(handler)
    session = accounts.session(token)
    if session and session.get('studio_admin'):
        session['user'] = {
            'id': 'studio:' + digest(token)[:24], 'name': 'Administrador',
            'username': 'admin', 'role': 'admin', 'studio_admin': True, 'local': bool(local),
        }
        return session
    if local:
        return local
    return guest_session(session, token) if session else None


# Compara el token de mutación con el CSRF de la sesión mediante comparación constante.
def csrf_valid(handler, session):
    return session and hmac.compare_digest(handler.headers.get('X-LSM-Token', ''), session['csrf'])


def studio_access_post(handler, route, payload, accounts, runtime, session):
    token = cookie_token(handler)
    if route == '/api/admin/unlock':
        runtime.admin.verify(payload.get('password'))
        issued, _ = accounts.issue(previous=token, studio_admin=True)
        close_user_cameras(runtime, session['user']['id'])
        set_session(handler, issued)
    else:
        if not session['user'].get('studio_admin'):
            raise PermissionError('El modo administrador ya está cerrado.')
        accounts.logout(token)
        close_user_cameras(runtime, session['user']['id'])
        issued, _ = accounts.issue()
        set_session(handler, issued)
    return {'ok': True}


# Procesa registro, acceso e invitaciones del modo autenticado; los rechaza en modo local.
def auth_post(handler, route, payload, accounts, runtime, session):
    if local_session(handler):
        raise PermissionError('Las cuentas están desactivadas en modo local.')
    user = session['user']
    token = cookie_token(handler)
    if route in ('/api/auth/setup', '/api/auth/register', '/api/auth/login'):
        if user and not user.get('guest'):
            raise ValueError('Cierra la sesión actual para cambiar de cuenta.')
        accounts.throttle('network:' + handler.client_address[0], limit=200)
        identity = 'account:' + str(payload.get('username', ''))[:40].lower()
        accounts.throttle(identity)
        if route == '/api/auth/login':
            identifier = accounts.login(payload.get('username'), payload.get('password'))['id']
        else:
            if payload.get('shared_consent') is not True:
                raise ValueError('Confirma que tus ejemplos guardados se compartirán con el equipo.')
            bootstrap = route == '/api/auth/setup'
            if bootstrap and (getattr(handler.server, 'public_origin', None) or handler.client_address[0] not in ('127.0.0.1', '::1')):
                raise PermissionError('Crea la primera cuenta desde localhost antes de publicar el servidor.')
            identifier = accounts.register(payload.get('username'), payload.get('name'), payload.get('password'), payload.get('invite'), bootstrap)
        accounts.clear_throttle(identity)
        issued, _ = accounts.issue(identifier, previous=token)
        set_session(handler, issued)
        return status_after_login(accounts, issued)
    if not user:
        raise PermissionError('Inicia sesión para continuar.')
    if route == '/api/auth/logout':
        accounts.logout(token)
        close_user_cameras(runtime, user['id'])
        issued, _ = accounts.issue()
        set_session(handler, issued)
        return status_after_login(accounts, issued)
    if user['role'] != 'admin' or user.get('studio_admin'):
        raise PermissionError('Esta opción requiere una cuenta administradora del equipo.')
    if route == '/api/auth/invite':
        return {'invite': accounts.invite(user['id']), 'expires_in_days': 7}
    if route == '/api/auth/users':
        return {'users': accounts.users()}
    if route == '/api/auth/enable':
        accounts.enable(payload.get('user_id'))
        return {'enabled': True}
    if route == '/api/auth/disable':
        accounts.disable(payload.get('user_id'), user['id'])
        close_user_cameras(runtime, payload.get('user_id'))
        return {'disabled': True}
    raise ValueError('Opción de cuenta no encontrada.')


# Devuelve la identidad y el CSRF nuevos después de cambiar de sesión.
def status_after_login(accounts, token):
    session = accounts.session(token)
    return {'user': session['user'], 'csrf': session['csrf'], 'setup_required': not accounts.configured(), 'shared': True}


# Consulta la procedencia de una toma pendiente o su recibo para verificar su propietario.
def draft_owner(language, identifier):
    validate_identifier(identifier)
    root = draft_dir(language)
    path = root / f'{identifier}.npz'
    if path.is_file():
        return sequence_metadata(path).get('owner_id')
    receipt = root / f'{identifier}.saved.json'
    if receipt.is_file():
        return json.loads(receipt.read_text(encoding='utf-8')).get('owner_id')
    raise ValueError('No existe esa toma pendiente.')


# Comprueba roles, consentimiento y propiedad de cámara o toma antes de una operación.
def authorize(runtime, user, route, payload):
    if (route in STUDIO_ROUTES or route.startswith(('/api/dataset/', '/api/models/'))) and not user.get('studio_admin'):
        raise PermissionError('Abre el modo administrador para usar esta opción.')
    if route in CAMERA_ROUTES:
        item = runtime.session(payload.get('session'))
        if item.owner_id != user['id']:
            raise PermissionError('La cámara pertenece a otra cuenta.')
    if route in ('/api/capture/save', '/api/capture/discard'):
        owner = draft_owner(payload.get('language'), payload.get('draft_id'))
        if owner != user['id'] and user['role'] != 'admin':
            raise PermissionError('Esta toma pendiente pertenece a otra cuenta.')
    if route == '/api/session' and any(s.owner_id == user['id'] and time.monotonic()-s.touched < 15 for s in runtime.sessions.values()):
        raise ValueError('Hay otra sesión de cámara enviando imágenes. Apágala en la otra pestaña; si ya la cerraste, espera 15 segundos y vuelve a activar la cámara.')
    if route == '/api/session' and payload.get('shared_consent') is not True:
        raise ValueError('Acepta el procesamiento en el servidor antes de activar la cámara.')


# Adjunta identidad y CSRF y calcula únicamente los datos que el usuario puede ver.
def shared_state(runtime, user, csrf):
    state = runtime.state(include_studio=bool(user.get('studio_admin')))
    state.update(token=csrf, user=user, shared=not user.get('local', False), api_version=3)
    return state
