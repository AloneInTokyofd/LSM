import base64
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import cv2
import sign_languages as langs
import web_dataset as data
import web_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(langs, 'WEB_DATA', self.root), patch.object(langs, 'WEB_MODELS', self.root / 'models')]
        for item in self.patches:
            item.start()
        self.points = np.ones((data.SEQUENCE_LENGTH, data.FEATURE_DIM), dtype=np.float32)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_seven_distinct_profiles_and_non_latin_labels(self):
        self.assertEqual(len({x['id'] for x in langs.LANGUAGES}), 7)
        self.assertEqual(langs.labels_for('jsl')[0]['text'], 'こんにちは')
        self.assertEqual(langs.labels_for('csl')[0]['text'], '你好')
        with self.assertRaises(ValueError):
            langs.dataset_dir('../lsm')

    def test_samples_do_not_cross_languages(self):
        data.save_sequence('lsm', 'hola', self.points)
        self.assertEqual(data.inventory('lsm')[0]['samples'], 1)
        self.assertEqual(data.inventory('asl')[0]['samples'], 0)

    def test_reject_incomplete_empty_and_nonfinite_sequences(self):
        for points in (self.points[:2], np.zeros_like(self.points), np.full_like(self.points, np.nan)):
            with self.assertRaises(ValueError):
                data.save_sequence('lsm', 'hola', points)
        self.assertEqual(data.inventory('lsm')[0]['samples'], 0)

    def test_incompatible_pipeline_is_not_counted(self):
        path = langs.dataset_dir('dgs') / 'hola'
        path.mkdir(parents=True)
        np.savez(path / 'bad.npz', points=self.points, pipeline=json.dumps({'version': 'old'}))
        self.assertEqual(data.inventory('dgs')[0]['invalid'], 1)
        self.assertEqual(data.inventory('dgs')[0]['samples'], 0)

    def test_training_requires_two_populated_complete_classes(self):
        with patch.object(data, 'inventory', return_value=[{'samples':29, 'text':'A'}]):
            with self.assertRaisesRegex(ValueError, '30 ejemplos'):
                data.read_dataset('lsm', langs.MIN_SAMPLES)
        data.save_sequence('lsm', 'hola', self.points)
        with self.assertRaises(ValueError):
            data.read_dataset('lsm', langs.MIN_SAMPLES)
        for label in ('hola', 'gracias'):
            for _ in range(langs.MIN_SAMPLES):
                data.save_sequence('asl', label, self.points)
        X, y, labels = data.read_dataset('asl', langs.MIN_SAMPLES)
        self.assertEqual(X.shape[0], 2 * langs.MIN_SAMPLES)
        train, val, test = data.split_per_class(y)
        self.assertEqual(tuple(map(len, (train, val, test))), (44, 8, 8))
        self.assertFalse(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        self.assertEqual(set(train) | set(val) | set(test), set(range(2 * langs.MIN_SAMPLES)))
        for part in (train, val, test):
            self.assertEqual(set(y[part]), {0, 1})

    def test_split_boundaries_and_existing_dataset_partition(self):
        for count, expected in ((4, None), (5, (6, 2, 2)), (6, (8, 2, 2)),
                                (7, (6, 4, 4)), (30, (44, 8, 8)), (42, (60, 12, 12))):
            with self.subTest(count=count):
                y = np.repeat([0, 1], count)
                if expected is None:
                    with self.assertRaises(ValueError):
                        data.split_per_class(y)
                    continue
                parts = data.split_per_class(y)
                self.assertEqual(tuple(map(len, parts)), expected)
                self.assertEqual(len(set(np.concatenate(parts))), len(y))
                for part in parts:
                    self.assertEqual(set(y[part]), {0, 1})

    def test_model_language_and_pipeline_must_match(self):
        version = 'a' * 32
        path = langs.model_dir('asl') / version
        path.mkdir(parents=True)
        (path.parent / 'current.json').write_text(json.dumps({'version': version}))
        (path / 'metadata.json').write_text(json.dumps({'language': 'lsm', 'pipeline': langs.pipeline_spec()}))
        with self.assertRaises(ValueError):
            web_app.model_metadata('asl')
        self.assertIsNone(web_app.model_metadata('jsl'))

    def test_camera_captures_complete_sequence_in_selected_language(self):
        runtime = web_app.Runtime()
        sid = runtime.new_session('libras')
        session = runtime.session(sid)
        session.holistic = object()
        session.capture = {'points': [], 'hands': 0}
        fake_utils = types.SimpleNamespace(mediapipe_detection=lambda image, model: (image, object()),
                                           extract_keypoints=lambda result: self.points[0], hands_detected=lambda result: True)
        image = cv2.imencode('.jpg', np.zeros((16, 16, 3), np.uint8))[1].tobytes()
        with patch.dict('sys.modules', {'utils': fake_utils}):
            for i in range(data.SEQUENCE_LENGTH):
                result = runtime.frame({'session': sid, 'image': base64.b64encode(image).decode(), 'mode': 'train'})
                if i < data.SEQUENCE_LENGTH - 1:
                    self.assertIsNone(result['draft_id'])
            self.assertIsNotNone(result['draft_id'])
        self.assertEqual(data.inventory('libras')[0]['samples'], 0)
        self.assertEqual(len(data.pending_drafts('libras')), 1)
        data.commit_draft('libras', result['draft_id'], 'Olá')
        self.assertEqual(data.inventory('libras')[0]['samples'], 1)
        self.assertEqual(data.inventory('lsm')[0]['samples'], 0)
        self.assertIsNone(session.capture)

    def test_name_after_recording_creates_persistent_language_specific_label(self):
        identifier = data.save_draft('jsl', self.points)
        self.assertEqual(sum(x['samples'] for x in data.inventory('jsl')), 0)
        result = data.commit_draft('jsl', identifier, '水')
        self.assertEqual(result['label']['text'], '水')
        self.assertEqual(langs.custom_labels('jsl')[0]['text'], '水')
        self.assertEqual(langs.custom_labels('lsm'), [])
        self.assertEqual(data.inventory('jsl')[-1]['samples'], 1)
        self.assertEqual(data.pending_drafts('jsl'), [])
        # Repetir la solicitud después de una respuesta perdida es idempotente.
        self.assertEqual(data.commit_draft('jsl', identifier, '水'), result)
        self.assertEqual(data.inventory('jsl')[-1]['samples'], 1)
        another = data.save_draft('jsl', self.points)
        data.commit_draft('jsl', another, '  水  ')
        self.assertEqual(len(langs.custom_labels('jsl')), 1)
        self.assertEqual(data.inventory('jsl')[-1]['samples'], 2)

    def test_existing_names_reuse_default_label(self):
        identifier = data.save_draft('lsm', self.points)
        result = data.commit_draft('lsm', identifier, '  HOLA  ')
        self.assertEqual(result['label']['id'], 'hola')
        self.assertEqual(langs.custom_labels('lsm'), [])
        self.assertEqual(data.inventory('lsm')[0]['samples'], 1)

    def test_invalid_name_keeps_draft_and_other_language_cannot_claim_it(self):
        identifier = data.save_draft('dgs', self.points)
        for name in ('', '   ', 'a' * 81, 'abc\nxyz', None):
            with self.assertRaises(ValueError):
                data.commit_draft('dgs', identifier, name)
        self.assertEqual(len(data.pending_drafts('dgs')), 1)
        self.assertEqual(langs.custom_labels('dgs'), [])
        with self.assertRaises(ValueError):
            data.commit_draft('lsm', identifier, 'Agua')
        with self.assertRaises(ValueError):
            data.commit_draft('dgs', '../../unsafe', 'Agua')
        data.discard_draft('dgs', identifier)
        self.assertEqual(data.pending_drafts('dgs'), [])
        self.assertTrue((data.draft_dir('dgs') / 'discarded' / f'{identifier}.npz').exists())

    def test_custom_labels_are_automatically_in_training_dataset(self):
        for name in ('Wasser', 'Brot'):
            label = langs.register_label('dgs', name)
            for _ in range(langs.MIN_SAMPLES):
                data.save_sequence('dgs', label['id'], self.points)
        X, y, labels = data.read_dataset('dgs', langs.MIN_SAMPLES)
        self.assertEqual({x['text'] for x in labels}, {'Wasser', 'Brot'})
        self.assertEqual(len(X), 2 * langs.MIN_SAMPLES)
        self.assertEqual(set(y), {0, 1})

    def test_no_model_means_no_prediction(self):
        runtime = web_app.Runtime()
        self.assertIsNone(runtime.load_model('jsl'))
        with self.assertRaises(ValueError):
            runtime.start_training('jsl', 40)

    def test_consecutive_capture_keeps_multiple_unnamed_takes(self):
        runtime = web_app.Runtime()
        sid = runtime.new_session('dgs')
        first = data.save_draft('dgs', self.points)
        with self.assertRaises(ValueError):
            runtime.start_capture(sid)
        self.assertTrue(runtime.start_capture(sid, consecutive=True)['capturing'])
        with self.assertRaises(ValueError):
            runtime.start_capture(sid, consecutive=True)
        # Una toma completa queda pendiente; iniciar la siguiente conserva ambas.
        runtime.session(sid).capture = None
        second = data.save_draft('dgs', self.points)
        self.assertTrue(runtime.start_capture(sid, consecutive=True)['capturing'])
        self.assertEqual({d['id'] for d in data.pending_drafts('dgs')}, {first, second})
        runtime.session(sid).capture = None  # Equivale a detener una toma parcial.
        data.commit_draft('dgs', first, 'Wasser')
        data.commit_draft('dgs', second, 'Wasser')
        self.assertEqual(data.inventory('dgs')[-1]['samples'], 2)
        self.assertEqual(data.pending_drafts('lsm'), [])

    def test_consecutive_capture_rejects_invalid_option_and_excess_pending(self):
        runtime = web_app.Runtime()
        sid = runtime.new_session('lsm')
        for option in ('yes', 1, None):
            with self.assertRaises(ValueError):
                runtime.start_capture(sid, consecutive=option)
        with patch.object(web_app, 'pending_drafts', return_value=[{}] * 100):
            with self.assertRaisesRegex(ValueError, '100 tomas'):
                runtime.start_capture(sid, consecutive=True)


if __name__ == '__main__':
    unittest.main()
