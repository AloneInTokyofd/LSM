"""Pruebas del ciclo de guardado con datos sintéticos exclusivamente temporales."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import sign_languages as langs
import web_train


class TrainingTests(unittest.TestCase):
    def test_candidate_cannot_replace_a_better_validation_model(self):
        reference = {'accuracy':1., 'macro_f1':1., 'accepted_accuracy':1., 'coverage':1.}
        self.assertTrue(web_train.candidate_is_not_worse(reference, reference))
        for key in reference:
            worse = {**reference, key:.9}
            self.assertFalse(web_train.candidate_is_not_worse(worse, reference))
        self.assertTrue(web_train.candidate_is_not_worse(reference, None))

    def test_training_publishes_complete_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.environ['MPLCONFIGDIR'] = str(root / 'cache')
            import tensorflow as tf
            import model
            from web_app import Runtime
            from config import SEQUENCE_LENGTH, FEATURE_DIM

            def tiny_model(num_classes, input_dim=FEATURE_DIM):
                result = tf.keras.Sequential([
                    tf.keras.layers.Input((SEQUENCE_LENGTH, input_dim)),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(num_classes, activation='softmax'),
                ])
                result.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                return result

            points = np.ones((24, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            targets = np.array([0] * 12 + [1] * 12)
            points[12:] *= .25
            labels = langs.labels_for('asl')[:2]
            with patch.object(langs, 'WEB_MODELS', root / 'models'), \
                 patch.object(web_train, 'WEB_DATA', root / 'data'), \
                 patch.object(web_train, 'read_dataset', return_value=(points, targets, labels)), \
                 patch.object(model, 'build_web_model', side_effect=tiny_model), \
                 contextlib.redirect_stdout(io.StringIO()):
                web_train.train_language('asl', 1, 'b' * 32)
                runtime = Runtime()
                loaded, metadata = runtime.load_model('asl')
                self.assertEqual(metadata['language'], 'asl')
                self.assertEqual(metadata['preprocessing'], 'body-hand-relative-v2')
                self.assertEqual(loaded.input_shape[-1], FEATURE_DIM + 128)
                self.assertTrue(metadata['activated'])
                self.assertTrue(metadata['augmentation']['training_only'])
                self.assertEqual(metadata['augmentation']['variants_per_original'], 8)
                self.assertEqual(metadata['augmentation']['camera_roll_degrees'], [4, 10])
                self.assertEqual(len(metadata['labels']), 2)
                self.assertEqual(sum(metadata['split'].values()), 24)
                self.assertEqual(loaded.output_shape[-1], 2)
                self.assertIsNone(runtime.load_model('lsm'))
                progress = json.loads((root / 'data' / 'jobs' / ('b' * 32 + '.json')).read_text())
                self.assertEqual(progress['state'], 'completed')

    def test_desktop_loader_skips_empty_and_partial_folders(self):
        import train
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'hola' / '0').mkdir(parents=True)
            (root / 'hola' / '1').mkdir(parents=True)
            (root / 'gracias' / '0').mkdir(parents=True)
            for i in range(train.SEQUENCE_LENGTH):
                np.save(root / 'hola' / '0' / f'frame_{i:03d}.npy', np.ones(train.FEATURE_DIM))
            np.save(root / 'gracias' / '0' / 'frame_000.npy', np.ones(train.FEATURE_DIM))
            with patch.object(train, 'DATA_PATH', str(root)), \
                 patch.object(train, 'ACTIONS', np.array(['hola', 'gracias'])), \
                 patch.object(train, 'NUM_SEQUENCES', 2), contextlib.redirect_stdout(io.StringIO()):
                X, y, _ = train.load_dataset()
            self.assertEqual(len(X), 1)
            self.assertEqual(y.tolist(), [0])


if __name__ == '__main__':
    unittest.main()
