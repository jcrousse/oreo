"""
reads the IMDB data from a CSV and generates a TF dataset.
Tokenization handled by SpaCy.

"""
from tqdm import tqdm

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path

from data.data_config import IMDB_CSV_SHUFFLE, LABELS, LABEL_COL, ID_COL, TEXT_COL, ENCODING


class NaiveTokenizer(list):
    def __init__(self, *args, **_):
        self.text = args[0]
        super().__init__(args[0].split(" "))
        self.sents = self._sents()

    def _sents(self):
        for sentence in self.text.split("."):
            yield NaiveTokenizer(sentence)


def identity(x):
    return x


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TextDataSetPrep:
    def __init__(self,
                 csv_path=IMDB_CSV_SHUFFLE,
                 labels=LABELS,
                 label_col=LABEL_COL,
                 id_col=ID_COL,
                 text_col=TEXT_COL,
                 encoding=ENCODING,
                 spacy_model='en_core_web_sm',
                 nrows=None,
                 chunksize=None,
                 ):

        self.csv_path = csv_path
        self.labels = labels
        self.label_col = label_col
        self.text_col = text_col
        self.id_col = id_col
        self.columns = [self.id_col, self.label_col]

        if csv_path is not None:
            label_df = pd.read_csv(self.csv_path, encoding=encoding, nrows=nrows, usecols=self.columns)
            self.label_df = label_df.loc[label_df[label_col].isin(self.labels), self.columns]
            self.text_df_gen = \
                pd.read_csv(self.csv_path,
                            encoding=encoding,
                            chunksize=chunksize,
                            usecols=[id_col, text_col, label_col],
                            nrows=nrows)
            if chunksize is None:
                self.text_df_gen = [self.text_df_gen]

            # self.label_values = self.label_df[self.label_col].unique()

            dataset_head = pd.read_csv(self.csv_path, encoding=encoding, nrows=5)
            self.tf_col_indices = sorted([list(dataset_head.columns).index(col)
                                          for col in [self.text_col, self.id_col, self.label_col]])

        if spacy_model is not None:
            nlp = spacy.load(spacy_model, disable=["tagger", "parser", "ner"])
            sentencizer = nlp.create_pipe("sentencizer")
            nlp.add_pipe(sentencizer)
            self.spacy_nlp = nlp
            self.tokenizer = self.spacy_nlp
        else:
            self.spacy_nlp = None
            self.tokenizer = NaiveTokenizer

        self.label_table = self._prepare_labels_lookup()

    def get_tokens_dataset(self,
                       n_per_label=None,
                       dataset_split=None,
                       seed=None,
                       tfr_names=None):

        selected_ids = self._selected_ids(n_per_label=n_per_label, seed=seed)
        dataset_ids = self._get_ids_per_dataset(selected_ids, dataset_split)
        dataset_list = []

        if tfr_names is None:
            tfr_names = [f"TFR.tfrecord"]

        assert len(tfr_names) == len(dataset_ids)

        if all(Path(tfr_name).exists() for tfr_name in tfr_names):
            dataset_list = [tf.data.TFRecordDataset(tfr_name) for tfr_name in tfr_names]
        else:
            for id_list, tfr_name in zip(dataset_ids, tfr_names):
                for text_df_chunk in self.text_df_gen:
                    try:
                        text_df_select = text_df_chunk.set_index(self.id_col).loc[id_list]
                    except KeyError:
                        text_df_select = pd.DataFrame()
                    if text_df_select.shape[0] > 0:
                        text_df_select[self.id_col] = text_df_select.index.values
                        text_df = text_df_select[~text_df_select[self.text_col].isnull()]
                        serialized_records = text_df.apply(
                            lambda x: self._serialize_tokens_tfr(x, False, False),
                            axis=1)
                        with tf.io.TFRecordWriter(tfr_name) as writer:
                            for serialized_item in tqdm(serialized_records.values):
                                writer.write(serialized_item.SerializeToString())
                dataset_raw = tf.data.TFRecordDataset(tfr_name)
                dataset = dataset_raw.map(self._deserialize_tokens)
                dataset_list.append(dataset)

        return dataset_list

    def get_ragged_tensors_dataset(self,
                                   split_sentences=False,
                                   split_characters=False,
                                   ):
        """
        https://www.tensorflow.org/tutorials/load_data/tfrecord does not seem to support nested features for now.
        Alternatively, we load all the text in memory and create one big RaggedTensor here.
        """

        text = []
        labels = []
        tqdm.pandas()
        for df in self.text_df_gen:
            text.extend(
                list(
                    df[self.text_col].progress_apply(
                        lambda x: self._text_split(x, split_sentences, split_characters))))
            labels.extend(list(df[self.label_col].values))

        ragged_text = tf.ragged.constant(text)
        labels = tf.constant(labels)
        # ragged_dataset = tf.data.Dataset.from_tensor_slices((ragged_text, labels))
        # ragged_dataset = self._encode_labels(ragged_dataset)
        encoded_labels = tf.one_hot(self.label_table.lookup(labels), len(self.labels))
        return ragged_text, encoded_labels

    def _selected_ids(self, n_per_label=1000, seed=None):
        def min_row(val1, val2):
            return val2 if val2 is None or 0 < val2 < val1 else val1
        return self.label_df.groupby(self.label_col, group_keys=False)\
                   .apply(lambda x: x.sample(min_row(len(x), n_per_label), random_state=seed))[self.id_col]

    def _text_split(self, text, split_sentences=False, split_characters=False):
        """
        splits a given text into tokens, and optionally in sentences, and characters.
        Returns (nested) list of (sentences) tokens (characters).
        """
        tokens = []
        tokenized_text = self.tokenizer(text)
        sentence_gen = tokenized_text.sents
        if not split_sentences:
            sentence_gen = [tokenized_text]

        def char_splitter(s):
            return [c for c in s]

        if not split_characters:
            char_splitter = identity

        for sentence in sentence_gen:
            tokens.append([char_splitter(str(t)) for t in sentence])

        if not split_sentences:
            tokens = tokens[0]

        return tokens

    @staticmethod
    def _get_ids_per_dataset(selected_ids, dataset_split=None, seed=None):
        """
        randomly selects IDs from possible vales 'selected_ids' following a list of proportions.
        e.g. for [0.5,0.25,0.25], returns three list of ids covering 50%, 25% and 25% of the selected_ids.
        """
        split_props = dataset_split or [1.0]
        assert (sum(split_props) == 1)

        ids_per_dataset = [[]] * len(split_props)
        remaining_ids = selected_ids
        for idx, split_prop in enumerate(split_props):
            if split_prop == 1.0:
                ids_per_dataset[idx] = remaining_ids
            else:
                ids_per_dataset[idx], remaining_ids = train_test_split(remaining_ids,
                                                                       test_size=1-split_prop,
                                                                       random_state=seed)
            if idx < len(split_props):
                split_props[idx+1:] = [e/sum(split_props[idx+1:]) for e in split_props[idx+1:]]

        return ids_per_dataset

    def _serialize_tokens_tfr(self, row, split_sent, split_char):
        text_data = self._text_split(row[self.text_col], split_sent, split_char)
        label = row[self.label_col]
        doc_id = row[self.id_col]

        context_features = tf.train.Features(feature={
            'doc_id': bytes_feature(doc_id),
            'label': bytes_feature(label)
        })

        feature_list = {}
        if text_data is not None:
            tokens_features = []
            for elem in text_data:
                tokens_features.append(bytes_feature(elem))

            feature_list['tokens'] = tf.train.FeatureList(feature=tokens_features)

        sequence_features = tf.train.FeatureLists(feature_list=feature_list)

        sequence_example = tf.train.SequenceExample(
            context=context_features,
            feature_lists=sequence_features,
        )

        return sequence_example

    @staticmethod
    def _deserialize_tokens(observation):
        context_features = {
            "doc_id": tf.io.FixedLenFeature([], dtype=tf.string),
            "label": tf.io.FixedLenFeature([], dtype=tf.string),
        }

        sequence_features = {
            "tokens": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        }

        context, sequences = tf.io.parse_single_sequence_example(
            serialized=observation,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return {'doc_id': context['doc_id'], 'tokens': sequences['tokens']}, context['label']


    def _prepare_labels_lookup(self):
        """ Create a StaticHashTable to lookup label to put them into int"""
        init_label = tf.lookup.KeyValueTensorInitializer(
            self.labels,
            tf.range(tf.size(self.labels, out_type=tf.int64), dtype=tf.int64),
            key_dtype=tf.string,
            value_dtype=tf.int64,
        )

        label_table = tf.lookup.StaticHashTable(
            init_label,
            default_value=-1,
        )
        return label_table

    def _encode_labels(self, dataset):
        return dataset.map(lambda x, y: (x, tf.one_hot(self.label_table.lookup(y), len(self.labels))))
