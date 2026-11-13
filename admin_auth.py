"""Autorización local de operaciones administrativas; nunca guarda texto claro."""
import argparse
import getpass
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
import warnings
from pathlib import Path


ITERATIONS = 600000
LOCAL_CONFIG = Path(__file__).resolve().parent / 'web_data' / 'admin' / 'auth.json'
TEAM_CONFIG = Path(__file__).resolve().with_name('admin_default.json')


def configure_password(path, password, *, replace=False):
    """Guarda un verificador privado y conserva la configuración anterior si falla."""
    if not isinstance(password, str) or not 0 < len(password) <= 256:
        raise ValueError('La contraseña debe tener entre 1 y 256 caracteres.')
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError('Ya hay una contraseña local. Usa --replace si quieres cambiarla.')
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, ITERATIONS)
    config = {
        'algorithm': 'pbkdf2-sha256', 'iterations': ITERATIONS,
        'salt': salt.hex(), 'digest': digest.hex(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.admin-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(config, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# Protege las operaciones administrativas con contraseña derivada y límite de intentos.
class AdminGuard:
    # Inicializa la ruta de credenciales y los contadores de bloqueo temporal.
    def __init__(self, config_path, *, default_config=None):
        self.config_path = Path(config_path)
        self.default_config = Path(default_config) if default_config is not None else None
        self.failures = 0
        self.window_started = 0
        self.blocked_until = 0

    # Comprueba la contraseña contra PBKDF2; rechaza credenciales inválidas o intentos bloqueados.
    def verify(self, password):
        now = time.monotonic()
        if now < self.blocked_until:
            raise PermissionError('Demasiados intentos. Espera cinco minutos antes de volver a intentar.')
        if now - self.window_started > 300:
            self.failures = 0
            self.window_started = now
        valid_input = isinstance(password, str) and 0 < len(password) <= 256
        supplied = password.encode('utf-8') if valid_input else b''
        configured = False
        # La clave del equipo funciona en todas las copias que incluyen su verificador.
        # Las credenciales de una instalación se conservan como acceso adicional.
        if self.default_config is not None:
            team_valid = self._matches_config(self.default_config, supplied)
            configured = team_valid is not None
            if valid_input and team_valid:
                self._check(True, now)
                return
        secret = os.environ.get('SENA_ADMIN_PASSWORD')
        if secret:
            configured = True
            valid = valid_input and hmac.compare_digest(supplied, secret.encode('utf-8'))
        else:
            local_valid = self._matches_config(self.config_path, supplied)
            configured = configured or local_valid is not None
            valid = valid_input and bool(local_valid)
        if not configured:
            raise PermissionError(
                'Admin no está configurado en esta instalación. Si ejecutas Seña en tu equipo, '
                'ejecuta admin_auth.py desde Terminal. En un servidor compartido, solicita la configuración a quien lo administra.'
            )
        self._check(valid, now)

    @staticmethod
    def _matches_config(path, supplied):
        """Devuelve None si el verificador falta o está dañado; False si la clave difiere."""
        try:
            config = json.loads(path.read_text(encoding='utf-8'))
            if config['algorithm'] != 'pbkdf2-sha256' or config['iterations'] != ITERATIONS:
                return None
            salt = bytes.fromhex(config['salt'])
            expected = bytes.fromhex(config['digest'])
            if len(salt) != 16 or len(expected) != 32:
                return None
        except (OSError, ValueError, KeyError, TypeError):
            return None
        digest = hashlib.pbkdf2_hmac('sha256', supplied, salt, config['iterations'])
        return hmac.compare_digest(digest, expected)

    def _check(self, valid, now):
        if not valid:
            self.failures += 1
            if self.failures >= 5:
                self.blocked_until = now + 300
            raise PermissionError('Contraseña de administrador incorrecta.')
        self.failures = 0
        self.window_started = now



def main():
    parser = argparse.ArgumentParser(description='Configura la contraseña de Admin en esta copia local de Seña.')
    parser.add_argument('--replace', action='store_true', help='Reemplaza la contraseña local ya configurada.')
    args = parser.parse_args()
    if os.environ.get('SENA_ADMIN_PASSWORD'):
        parser.error('SENA_ADMIN_PASSWORD ya configura Admin en este entorno.')
    if LOCAL_CONFIG.exists() and not args.replace:
        parser.error('Ya hay una contraseña local. Usa --replace si quieres cambiarla.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            password = getpass.getpass('Contraseña acordada para Admin: ')
            confirmation = getpass.getpass('Confirma la contraseña: ')
        if password != confirmation:
            parser.error('Las contraseñas no coinciden.')
        configure_password(LOCAL_CONFIG, password, replace=args.replace)
    except (EOFError, getpass.GetPassWarning):
        parser.error('Abre una terminal interactiva para escribir la contraseña de forma oculta.')
    except KeyboardInterrupt:
        parser.exit(1, '\nConfiguración cancelada.\n')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('Admin configurado para esta instalación de Seña.')


if __name__ == '__main__':
    main()
