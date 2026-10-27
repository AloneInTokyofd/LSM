import unittest
import numpy as np
from config import FEATURE_DIM
from recognition_features import normalize_sequence, augment_training, augment_viewpoints, regions, validation_threshold, decision_metrics
from recognition_features import prepare_sequence, local_hands, fit_hand_guard, hand_guard_accept, feature_dimension, HAND_PREPROCESSING


class FeatureTests(unittest.TestCase):
    def sequence(self):
        sequence = np.zeros((30, FEATURE_DIM), dtype=np.float32)
        pose, _, left, _ = regions(sequence)
        pose[:, 11, :] = [.4, .5, .05, 1]
        pose[:, 12, :] = [.6, .5, .05, 1]
        left[:, :, :] = [.55, .7, .02]
        left[:, 0, 1] += np.linspace(0, .1, 30)
        return sequence

    def test_normalization_removes_translation_and_scale_but_retains_motion(self):
        original = self.sequence()
        translated = original.copy()
        for region in regions(translated):
            present = np.any(region[..., :3] != 0, axis=-1)
            region[..., :3] = np.where(present[..., None], region[..., :3] * .8, 0)
            region[..., :2] += np.where(present[..., None], [.1, .05], [0, 0])
        normalized = normalize_sequence(original)
        np.testing.assert_allclose(normalized, normalize_sequence(translated), atol=1e-6)
        self.assertNotEqual(regions(normalized)[2][0, 0, 1], regions(normalized)[2][-1, 0, 1])
        self.assertTrue(np.all(regions(normalized)[3] == 0))
        self.assertTrue(np.all(regions(normalized)[0][:, 11, 3] == 1))

    def test_augmentation_preserves_absence_labels_and_original_data(self):
        original = np.stack([normalize_sequence(self.sequence())] * 2)
        saved = original.copy()
        targets = np.array([[1, 0], [0, 1]], np.float32)
        varied, labels = augment_training(original, targets)
        self.assertEqual(len(varied), 6)
        np.testing.assert_array_equal(varied[:2], original)
        np.testing.assert_array_equal(original, saved)
        self.assertEqual(len(labels), len(varied))
        np.testing.assert_array_equal(labels, np.tile(targets, (3, 1)))
        self.assertTrue(np.all(regions(varied)[3] == 0))
        self.assertFalse(np.allclose(varied[2:], np.tile(original, (2, 1, 1))))

    def test_decision_filter_reports_rejections_without_inflating_accuracy(self):
        probs = np.array([[.99, .01], [.01, .99], [.6, .4], [.01, .99]])
        y = np.array([0, 1, 0, 0])
        result = decision_metrics(probs, y)
        self.assertEqual(result['accepted'], 3)
        self.assertAlmostEqual(result['accepted_accuracy'], 2 / 3)
        self.assertAlmostEqual(result['coverage'], .75)
        calibrated = validation_threshold(probs, y)
        self.assertEqual(calibrated['threshold'], .95)

    def test_shoulders_missing_has_finite_fallback(self):
        empty = np.zeros((30, FEATURE_DIM), np.float32)
        np.testing.assert_array_equal(normalize_sequence(empty), empty)

    def test_compact_network_produces_finite_class_probabilities(self):
        import tensorflow as tf
        from model import build_web_model
        model = build_web_model(3)
        probabilities = model(np.stack([normalize_sequence(self.sequence())]), training=False).numpy()
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0], atol=1e-6)
        self.assertLess(model.count_params(), 300000)

    def hand_sequence(self):
        sequence = self.sequence()
        hand = regions(sequence)[2]
        for point in range(21):
            hand[:, point, :] = [.5 + .008 * (point % 4), .7 - .01 * (point // 4), .001 * (point % 3)]
        hand[:, 0, :] = [.5, .7, 0]
        return sequence

    def test_local_hand_shape_is_invariant_to_translation_and_size(self):
        original = self.hand_sequence()
        moved = original.copy()
        hand = regions(moved)[2]
        hand[:] = hand * .75 + [.12, .1, .02]
        np.testing.assert_allclose(local_hands(original), local_hands(moved), atol=1e-5)
        prepared = prepare_sequence(original)
        self.assertEqual(prepared.shape, (30, FEATURE_DIM + 128))
        self.assertEqual(feature_dimension(HAND_PREPROCESSING), FEATURE_DIM + 128)
        self.assertTrue(np.all(prepared[:, -1] == 0))
        self.assertTrue(np.all(prepared[:, -2] == 1))

    def test_guard_accepts_similar_hand_shape_and_rejects_distant_shapes(self):
        known = self.hand_sequence()
        train = np.stack([known] * 6)
        labels = np.array([0, 0, 0, 1, 1, 1])
        guard = fit_hand_guard(train, labels, np.stack([known, known]), np.array([0, 1]))
        self.assertTrue(hand_guard_accept(known, 0, guard))
        other = known.copy()
        regions(other)[2][:, [4, 8, 12, 16, 20], 0] += 2
        self.assertFalse(hand_guard_accept(other, 0, guard))
        self.assertTrue(hand_guard_accept(other, 0, None), 'Los modelos anteriores no reciben un filtro inventado.')

    def test_new_features_keep_flags_and_missing_hand_zero_during_augmentation(self):
        prepared = prepare_sequence(self.hand_sequence())
        varied, labels = augment_training(prepared[None], np.array([[1, 0]], dtype=np.float32))
        self.assertEqual(varied.shape, (3, 30, FEATURE_DIM + 128))
        self.assertEqual(len(labels), 3)
        self.assertTrue(np.all(varied[:, :, FEATURE_DIM + 63:FEATURE_DIM + 126] == 0))
        self.assertTrue(np.all(varied[:, :, -2] == 1))

    def test_legacy_preprocessing_remains_compatible(self):
        points = self.sequence()
        np.testing.assert_array_equal(prepare_sequence(points, None), points)
        np.testing.assert_array_equal(prepare_sequence(points, 'body-relative-v1'), normalize_sequence(points))
        with self.assertRaises(ValueError):
            feature_dimension('unknown')

    def test_viewpoint_variants_preserve_labels_absences_depth_and_visibility(self):
        raw = np.stack([self.hand_sequence(), self.sequence()])
        saved = raw.copy()
        targets = np.eye(2, dtype=np.float32)
        varied, labels = augment_viewpoints(raw, targets)
        np.testing.assert_array_equal(raw, saved)
        np.testing.assert_array_equal(varied[:2], raw)
        np.testing.assert_array_equal(labels, np.tile(targets, (3, 1)))
        self.assertTrue(np.isfinite(varied).all())
        for variant in (varied[2:4], varied[4:6]):
            self.assertFalse(np.allclose(variant, raw))
            for before, after in zip(regions(raw), regions(variant)):
                np.testing.assert_array_equal(before[..., 2:], after[..., 2:])
                absent = np.all(before[..., :3] == 0, axis=-1)
                self.assertTrue(np.all(after[absent] == 0))
        # The expanded shape guard should accept these training viewpoints.
        guard = fit_hand_guard(varied, np.argmax(labels, axis=1), raw, np.array([0, 1]))
        for points, label in zip(varied, np.argmax(labels, axis=1)):
            self.assertTrue(hand_guard_accept(points, label, guard))

    def test_viewpoint_rotation_is_constant_through_time(self):
        raw = self.hand_sequence()
        raw[:] = raw[:1]
        augmented, _ = augment_viewpoints(raw[None], np.array([0]))
        for sequence in augmented:
            np.testing.assert_allclose(sequence, np.repeat(sequence[:1], len(sequence), axis=0))


if __name__ == '__main__': unittest.main()
