from unittest import TestCase
import os

import tensorflow as tf

from data.dataset_prep import TextDataSetPrep
from data.data_config import IMDB_CSV_TEST


class TestTextDataSetPrep(TestCase):
    def setUp(self) -> None:
        self.temp_tfr_path = "tests/tfr_test.tfrecord"
        self.temp_pkl_path = "tests/pkl_test"
        self._cleanup()
        # if not os.path.isdir(self.temp_pkl_path):
        #     os.mkdir(self.temp_pkl_path)

    def tearDown(self) -> None:
        self._cleanup()

    def _cleanup(self):
        if os.path.isfile(self.temp_tfr_path):
            os.remove(self.temp_tfr_path)
        # if os.path.isdir(self.temp_pkl_path):
        #     rmtree(self.temp_pkl_path)

    def test_get_imdb_data(self):
        _ = TextDataSetPrep(nrows=10)

    def test_get_x_y(self):
        tds = TextDataSetPrep(nrows=40, spacy_model=None, csv_path=IMDB_CSV_TEST)
        df_ids = tds._selected_ids(seed=7357, n_per_label=2)
        self.assertIn('5855_1.txt', df_ids.values)

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

    def test_write_tfr(self):
        self._cleanup()
        _ = TextDataSetPrep(csv_path=IMDB_CSV_TEST).write_tfr_datasets(tfr_names=[self.temp_tfr_path])
        _ = TextDataSetPrep(csv_path=IMDB_CSV_TEST).write_tfr_datasets(tfr_names=[self.temp_tfr_path])

    def test_serial_deserial(self):
        tds = TextDataSetPrep(csv_path=None, id_col='id', text_col='text', label_col='label')
        data = {
            'id': 'abc',
            'label': 'label',
            'text': "this is my text for testing. It has two sentences"
        }
        serialized = tds._serialize_tokens_tfr(data)
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

    def test_doc_to_pickle(self):
        self._cleanup()
        tds = TextDataSetPrep(nrows=20)
        dataset_dir = tds.csv_to_pickle(target_dir=self.temp_pkl_path)
        first_id = tds.label_df.iloc[0, 0]
        first_label = tds.label_df.iloc[0, 1]
        test_path = os.path.join(dataset_dir, first_label, first_id + '.pkl')
        self.assertTrue(os.path.isfile(test_path))

    def test_split_all(self):
        tds = TextDataSetPrep(nrows=20)
        text = "This is my first sentence. This is Sparta."
        tokens, w_l, s_l = tds._split_all(text)
        self.assertListEqual([6, 4], s_l)
        self.assertEqual([4, 2, 2, 5, 8, 1, 4, 2, 6, 1], w_l)
