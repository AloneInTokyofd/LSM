import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import sign_languages as langs
import alphabets
import web_dataset as data

class AlphabetTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.patch = patch.object(langs, 'WEB_DATA', Path(self.folder.name))
        self.patch.start()
    def tearDown(self):
        self.patch.stop()
        self.folder.cleanup()
    def test_catalogs_are_distinct_and_cover_required_symbols(self):
        self.assertEqual(len(alphabets.alphabet_for('lsm')['symbols']), 27)
        self.assertIn('Ñ', alphabets.alphabet_for('lsm')['symbols'])
        self.assertIn('ß', alphabets.alphabet_for('dgs')['symbols'])
        self.assertEqual(len(alphabets.alphabet_for('jsl')['symbols']), 46)
        self.assertIn('ん', alphabets.alphabet_for('jsl')['symbols'])
        self.assertIn('ZH', alphabets.alphabet_for('csl')['symbols'])
        for language in langs.LANGUAGES:
            catalog = alphabets.alphabet_for(language['id'])
            values = catalog['symbols'] + catalog['extras']
            self.assertEqual(len(values), len(set(values)))
    def test_add_all_is_persistent_idempotent_and_does_not_claim_models(self):
        result = alphabets.add_all_alphabets()
        self.assertEqual(len(result), 7)
        self.assertTrue(all(item['added'] > 0 for item in result))
        again = alphabets.add_all_alphabets()
        self.assertTrue(all(item['added'] == 0 for item in again))
        for item in langs.LANGUAGES:
            rows = data.inventory(item['id'])
            status = alphabets.alphabet_status(item['id'], rows, None)
            self.assertTrue(all(letter['label_id'] for letter in status['symbols']))
            self.assertTrue(all(letter['samples'] == 0 and not letter['trained'] for letter in status['symbols']))
    def test_existing_letter_and_samples_are_preserved(self):
        original = langs.register_label('lsm', 'a')
        points = np.ones((data.SEQUENCE_LENGTH, data.FEATURE_DIM), dtype=np.float32)
        data.save_sequence('lsm', original['id'], points)
        result = alphabets.add_alphabet('lsm')
        self.assertEqual(result['added'], 26)
        row = alphabets.alphabet_status('lsm', data.inventory('lsm'), {'labels':[original]})['symbols'][0]
        self.assertEqual(row['label_id'], original['id'])
        self.assertEqual(row['samples'], 1)
        self.assertTrue(row['trained'])
        self.assertEqual(langs.custom_labels('asl'), [])
    def test_sharp_s_does_not_merge_with_ss_and_optional_variants_work(self):
        original = langs.register_label('dgs', 'SS')
        alphabets.add_alphabet('dgs')
        sharp_s = alphabets.find_symbol_label(langs.labels_for('dgs'), 'ß')
        self.assertNotEqual(original['id'], sharp_s['id'])
        alphabets.add_alphabet('jsl', extras=True)
        self.assertIsNotNone(alphabets.find_symbol_label(langs.labels_for('jsl'), 'が'))
        self.assertIsNone(alphabets.find_symbol_label(langs.labels_for('jsl'), 'あ'))
    def test_adding_empty_letters_does_not_block_training_collected_classes(self):
        alphabets.add_alphabet('asl')
        points = np.ones((data.SEQUENCE_LENGTH, data.FEATURE_DIM), dtype=np.float32)
        for symbol in ['A','B']:
            label = alphabets.find_symbol_label(langs.labels_for('asl'), symbol)
            for _ in range(langs.MIN_SAMPLES):
                data.save_sequence('asl', label['id'], points)
        X, y, labels = data.read_dataset('asl', langs.MIN_SAMPLES)
        self.assertEqual(len(X), 2 * langs.MIN_SAMPLES)
        self.assertEqual({row['text'] for row in labels}, {'A','B'})

if __name__ == '__main__': unittest.main()
