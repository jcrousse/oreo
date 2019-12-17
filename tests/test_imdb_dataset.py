from unittest import TestCase
import os

import tensorflow as tf

from data.dataset_prep import TextDataSetPrep


class TestTextDataSetPrep(TestCase):
    def setUp(self) -> None:
        self.temp_tfr_path = "tests/tfr_test.tfrecord"
        self._cleanup()

    def tearDown(self) -> None:
        self._cleanup()

    def _cleanup(self):
        if os.path.isfile(self.temp_tfr_path):
            os.remove(self.temp_tfr_path)

    def test_get_imdb_data(self):
        _ = TextDataSetPrep(nrows=10)

    def test_get_x_y(self):
        tds = TextDataSetPrep(nrows=10, spacy_model=None)
        df_ids = tds._selected_ids(seed=7357)
        self.assertIn('5839_3.txt', df_ids.values)

    def test_text_split(self):
        doc = "this is the first sentence. This is the second one. \n\n This is  a new paragraph"
        tds_def = TextDataSetPrep(nrows=1, spacy_model=None)
        tokens = tds_def._text_split(doc)
        self.assertEqual('this', tokens[0])
        self.assertEqual(len(tokens), 17)
        self.assertIsInstance(tokens[0], str)

        tokens_s = tds_def._text_split(doc, split_sentences=True)

        self.assertEqual('this', tokens_s[0][0])
        self.assertEqual(len(tokens_s), 3)
        self.assertIsInstance(tokens_s[0][0], str)

        tokens_s_c = tds_def._text_split(doc, split_sentences=True, split_characters=True)
        self.assertEqual('t', tokens_s_c[0][0][0])
        self.assertEqual(len(tokens_s_c), 3)
        self.assertIsInstance(tokens_s_c[0][0][0], str)

        tds_spacy = TextDataSetPrep(nrows=1)
        tokens_s = tds_spacy._text_split(doc, split_sentences=True)
        self.assertEqual('this', tokens_s[0][0])
        self.assertEqual(len(tokens_s), 3)
        self.assertIsInstance(tokens_s[0][0], str)

    def test_get_ids_per_dataset(self):
        v1 = TextDataSetPrep._get_ids_per_dataset(list(range(10)), [0.5, 0.5], seed=7357)
        v2 = TextDataSetPrep._get_ids_per_dataset(list(range(10)), [0.25, 0.25, 0.25, 0.25], seed=7357)
        self.assertEqual(len(v1), 2)
        self.assertEqual(len(v2), 4)
        self.assertListEqual([4, 1, 3, 5, 7], v1[0])

    def test_get_tf_dataset(self):
        self._cleanup()
        _ = TextDataSetPrep(chunksize=100).get_tf_dataset(tfr_names=[self.temp_tfr_path])
        _ = TextDataSetPrep(chunksize=100).get_tf_dataset(tfr_names=[self.temp_tfr_path])

    def test_serial_deserial(self):
        tds = TextDataSetPrep(csv_path=None, id_col='id', text_col='text', label_col='label')
        data = {
            'id': 'abc',
            'label': 'label',
            'text': "this is my text for testing. It has two sentences"
        }
        serialized = tds._serialize_tokens_tfr(data, False, False)
        with tf.io.TFRecordWriter(self.temp_tfr_path) as writer:
            writer.write(serialized.SerializeToString())
        dataset = tf.data.TFRecordDataset(self.temp_tfr_path)
        for e in dataset:
            _ = tds._deserialize_tokens(e)

    def test_ragged_memory(self):
        tds = TextDataSetPrep(nrows=100)
        x, _ = tds.get_ragged_tensors_dataset()
        self.assertListEqual(list(x.bounding_shape().numpy()), [100, 870])

        x, _ = tds.get_ragged_tensors_dataset(split_characters=True)
        self.assertListEqual(list(x.bounding_shape().numpy()), [100, 870, 64])

        x, _ = tds.get_ragged_tensors_dataset(split_characters=True, split_sentences=True)
        self.assertListEqual(list(x.bounding_shape().numpy()), [100, 32, 183, 64])
