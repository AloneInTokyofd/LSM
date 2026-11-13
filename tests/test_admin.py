"""Autorización y retirada de ejemplos: solo archivos de prueba temporales."""
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sign_languages as langs
import web_dataset as data
import web_app
from admin_auth import AdminGuard, configure_password


# Ayudante de prueba para configure.
def configure(path):
    salt = bytes.fromhex('0123456789abcdef0123456789abcdef')
    digest = hashlib.pbkdf2_hmac('sha256', b'test-only-password', salt, 600000)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'algorithm':'pbkdf2-sha256', 'iterations':600000,
                                'salt':salt.hex(), 'digest':digest.hex()}))


# Ayudante de prueba para AdminTests.
class AdminTests(unittest.TestCase):
    # Prepara el estado temporal aislado que necesita cada prueba.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(langs, 'WEB_DATA', self.root), patch.object(web_app, 'WEB_DATA', self.root), patch.object(langs, 'WEB_MODELS', self.root / 'models')]
        for item in self.patches:
            item.start()
        self.auth = self.root / 'admin' / 'auth.json'
        configure(self.auth)
        self.runtime = web_app.Runtime()
        self.points = np.ones((data.SEQUENCE_LENGTH, data.FEATURE_DIM), dtype=np.float32)
        self.file = data.save_sequence('lsm', 'hola', self.points)
        data.save_sequence('lsm', 'gracias', self.points)
        self.historical = self.root / 'asl' / 'historical.npz'
        self.historical.parent.mkdir(parents=True)
        self.historical.write_bytes(b'historical examples')
        self.payload = {'language':'lsm', 'label':'hola', 'password':'test-only-password', 'confirmed':True}

    # Elimina los recursos temporales creados durante la prueba.
    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    # Prueba automatizada: verifica missing wrong or unconfigured password cannot delete.
    def test_missing_wrong_or_unconfigured_password_cannot_delete(self):
        for password in (None, '', 'wrong', ['test-only-password']):
            with self.assertRaises(PermissionError):
                self.runtime.delete_recordings({**self.payload, 'password':password})
        self.assertEqual(data.inventory('lsm')[0]['samples'], 1)
        self.auth.unlink()
        with self.assertRaises(PermissionError):
            self.runtime.delete_recordings(self.payload)

    # Prueba automatizada: verifica rate limit and expiry.
    def test_rate_limit_and_expiry(self):
        guard = AdminGuard(self.auth)
        with patch('admin_auth.time.monotonic', return_value=1000):
            for _ in range(5):
                with self.assertRaises(PermissionError):
                    guard.verify('wrong')
            with self.assertRaisesRegex(PermissionError, 'Demasiados intentos'):
                guard.verify('test-only-password')
        with patch('admin_auth.time.monotonic', return_value=1301):
            guard.verify('test-only-password')

    # Prueba automatizada: verifica confirmation training and capture block deletion.
    def test_confirmation_training_and_capture_block_deletion(self):
        for confirmation in (False, None, 'true'):
            with self.assertRaises(ValueError):
                self.runtime.delete_recordings({**self.payload, 'confirmed':confirmation})
        with patch.object(self.runtime, 'training', return_value=True):
            with self.assertRaises(ValueError):
                self.runtime.delete_recordings(self.payload)
        self.runtime.sessions['test'] = types.SimpleNamespace(capture={'points':[]})
        with self.assertRaises(ValueError):
            self.runtime.delete_recordings(self.payload)
        self.assertTrue((langs.dataset_dir('lsm') / 'hola' / self.file).is_file())

    # Prueba automatizada: verifica deletion is recoverable scoped and updates inventory.
    def test_deletion_is_recoverable_scoped_and_updates_inventory(self):
        original = langs.dataset_dir('lsm') / 'hola' / self.file
        before = original.read_bytes()
        original.with_name('invalid.npz').write_bytes(b'invalid sequence')
        data.inventory('lsm')  # Prime cache before deletion.
        result = self.runtime.delete_recordings(self.payload)
        self.assertEqual(result['deleted'], 2)
        archived = self.root / 'trash' / 'lsm' / result['archive_id']
        self.assertEqual((archived / 'hola' / self.file).read_bytes(), before)
        self.assertEqual(json.loads((archived / 'deletion.json').read_text())['label'], 'hola')
        self.assertEqual(data.inventory('lsm')[0]['samples'], 0)
        self.assertEqual(data.inventory('lsm')[1]['samples'], 1)
        self.assertEqual(self.historical.read_bytes(), b'historical examples')
        self.assertIn('hola', [label['id'] for label in langs.labels_for('lsm')])
        with self.assertRaises(ValueError):
            self.runtime.delete_recordings(self.payload)

    # Prueba automatizada: verifica invalid label and archive write failure leave data intact.
    def test_invalid_label_and_archive_write_failure_leave_data_intact(self):
        with self.assertRaises(ValueError):
            self.runtime.delete_recordings({**self.payload, 'label':'../../outside'})
        with patch.object(Path, 'write_text', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.runtime.delete_recordings(self.payload)
        self.assertEqual(data.inventory('lsm')[0]['samples'], 1)

    # Prueba automatizada: verifica deleting a trained sign disables model and clears cached predictions.
    def test_deleting_a_trained_sign_disables_model_and_clears_cached_predictions(self):
        root = langs.model_dir('lsm')
        version = 'a' * 32
        bundle = root / version
        bundle.mkdir(parents=True)
        (root / 'current.json').write_text(json.dumps({'version':version}))
        (bundle / 'metadata.json').write_text(json.dumps({'language':'lsm', 'pipeline':langs.pipeline_spec(),
                                                        'labels':langs.labels_for('lsm')[:2]}))
        (bundle / 'model.keras').write_bytes(b'not loaded in this test')
        self.assertIsNotNone(web_app.model_metadata('lsm'))
        self.runtime.models['lsm'] = ('cached', object(), {})
        result = self.runtime.delete_recordings(self.payload)
        self.assertTrue(result['model_invalidated'])
        self.assertIsNone(web_app.model_metadata('lsm'))
        self.assertIsNone(self.runtime.load_model('lsm'))
        self.assertNotIn('lsm', self.runtime.models)
        self.assertTrue((bundle / 'model.keras').is_file())
        self.assertTrue(json.loads((root / 'current.json').read_text())['requires_retraining'])
        profile = next(p for p in self.runtime.state()['profiles'] if p['id'] == 'lsm')
        self.assertIsNone(profile['model'])
        self.assertIn('desactivado', profile['model_error'])

    # Prueba automatizada: verifica selected training preserves incomplete unselected examples.
    def test_selected_training_preserves_incomplete_unselected_examples(self):
        for label in ('hola', 'gracias'):
            for _ in range(29):
                data.save_sequence('lsm', label, self.points)
        data.save_sequence('lsm', 'adios', self.points)
        raw, _, labels = data.read_dataset('lsm', 30, ['hola', 'gracias'])
        self.assertEqual(len(raw), 60)
        self.assertEqual({x['id'] for x in labels}, {'hola', 'gracias'})
        with self.assertRaises(ValueError):
            data.read_dataset('lsm', 30, ['hola', 'adios'])
        with self.assertRaises(ValueError):
            data.read_dataset('lsm', 30, ['hola', 'unknown'])
        self.assertEqual(next(x for x in data.inventory('lsm') if x['id'] == 'adios')['samples'], 1)

    # Prueba automatizada: verifica http route enforces both csrf and admin password.
    def test_http_route_enforces_both_csrf_and_admin_password(self):
        # Ayudante de prueba para request.
        def request(payload, token):
            handler = web_app.Handler.__new__(web_app.Handler)
            body = json.dumps(payload).encode()
            handler.path = '/api/admin/delete-recordings'
            handler.server = types.SimpleNamespace(server_port=8765)
            handler.headers = {'Host':'127.0.0.1:8765', 'Origin':'http://127.0.0.1:8765',
                               'Content-Length':str(len(body)), 'X-LSM-Token':token}
            handler.rfile = io.BytesIO(body)
            responses = []
            handler.send_data = lambda status, data: responses.append((status, data))
            with patch.object(web_app, 'RUNTIME', self.runtime), \
                 patch.object(web_app.team_access, 'authenticate', return_value={'user':{'id':'owner','role':'admin','studio_admin':True},'csrf':web_app.TOKEN}):
                handler.do_POST()
            return responses[0]
        self.assertEqual(request(self.payload, 'wrong-csrf')[0], 403)
        self.assertEqual(request({**self.payload, 'password':''}, web_app.TOKEN)[0], 403)
        self.assertEqual(request(self.payload, web_app.TOKEN)[0], 200)


class SharedAdminPasswordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.local = self.root / 'local.json'
        self.team = self.root / 'team.json'
        configure_password(self.team, 'shared-test-password')
        configure_password(self.local, 'local-test-password')
        self.environment = patch.dict('os.environ', {}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def test_shared_password_works_with_conflicting_local_configuration(self):
        original = self.local.read_bytes()
        guard = AdminGuard(self.local, default_config=self.team)
        guard.verify('shared-test-password')
        guard.verify('local-test-password')
        self.assertEqual(self.local.read_bytes(), original)

    def test_shared_password_works_with_environment_override(self):
        with patch.dict('os.environ', {'SENA_ADMIN_PASSWORD': 'environment-test-password'}):
            guard = AdminGuard(self.local, default_config=self.team)
            guard.verify('shared-test-password')
            guard.verify('environment-test-password')
            with self.assertRaises(PermissionError):
                guard.verify('local-test-password')

    def test_shared_password_survives_missing_or_damaged_local_config(self):
        for content in (None, '{broken', 'null', '{"algorithm":"unknown"}'):
            with self.subTest(content=content):
                if content is None:
                    self.local.unlink(missing_ok=True)
                else:
                    self.local.write_text(content)
                guard = AdminGuard(self.local, default_config=self.team)
                guard.verify('shared-test-password')
                with self.assertRaisesRegex(PermissionError, 'incorrecta'):
                    guard.verify('wrong')

    def test_standalone_environment_override_remains_exclusive(self):
        with patch.dict('os.environ', {'SENA_ADMIN_PASSWORD': 'environment-test-password'}):
            guard = AdminGuard(self.local)
            guard.verify('environment-test-password')
            for password in ('shared-test-password', 'local-test-password'):
                with self.assertRaises(PermissionError):
                    guard.verify(password)

    def test_rate_limit_applies_to_shared_password_and_expires(self):
        guard = AdminGuard(self.local, default_config=self.team)
        with patch('admin_auth.time.monotonic', return_value=1000):
            for _ in range(5):
                with self.assertRaises(PermissionError):
                    guard.verify('wrong')
            with self.assertRaisesRegex(PermissionError, 'Demasiados intentos'):
                guard.verify('shared-test-password')
        with patch('admin_auth.time.monotonic', return_value=1301):
            guard.verify('shared-test-password')
            self.assertEqual(guard.failures, 0)

    def test_all_malformed_verifiers_fail_closed(self):
        self.local.write_text('{}')
        self.team.write_text('null')
        with self.assertRaisesRegex(PermissionError, 'no está configurado'):
            AdminGuard(self.local, default_config=self.team).verify('shared-test-password')


if __name__ == '__main__':
    unittest.main()
