"""
reads the IMDB data from a CSV and generates a TF dataset.
Tokenization handled by SpaCy.

"""
import os
import pickle

import concurrent.futures

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

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


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _get_seq_feature(tokens, encoder=bytes_feature):
    tokens_features = []
    for elem in tokens:
        tokens_features.append(encoder(elem))
    return tokens_features


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

    def write_tfr_datasets(self,
                           n_per_label=None,
                           dataset_split=None,
                           seed=None,
                           tfr_names=None):

        selected_ids = self._selected_ids(n_per_label=n_per_label, seed=seed)
        dataset_ids = self._get_ids_per_dataset(selected_ids, dataset_split)
        # dataset_list = []

        if tfr_names is None:
            tfr_names = [f"TFR.tfrecord"]

        assert len(tfr_names) == len(dataset_ids)

        # if all(Path(tfr_name).exists() for tfr_name in tfr_names):
        #     dataset_list = [tf.data.TFRecordDataset(tfr_name) for tfr_name in tfr_names]
        # else:
        for id_list, tfr_name in zip(dataset_ids, tfr_names):
            for text_df_chunk in self.text_df_gen:
                try:
                    text_df_select = text_df_chunk.set_index(self.id_col).loc[id_list]
                except KeyError:
                    text_df_select = pd.DataFrame()
                if text_df_select.shape[0] > 0:
                    text_df_select[self.id_col] = text_df_select.index.values
                    text_df = text_df_select[~text_df_select[self.text_col].isnull()]

                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        # Start the load operations and mark each future with its URL
                        futures = [executor.submit(self._serialize_tokens_tfr, row)
                                   for row in tqdm(text_df.iterrows(), total=text_df.shape[0])]
                        for future in tqdm(concurrent.futures.as_completed(futures), total=text_df.shape[0]):
                            try:
                                data = future.result()
                                with tf.io.TFRecordWriter(tfr_name) as writer:
                                    writer.write(data.SerializeToString())
                            except Exception as exc:
                                print('%r generated an exception: %s' % (exc))

                    # tqdm.pandas()
                    # serialized_records = text_df.progress_apply(
                    #     lambda x: self._serialize_tokens_tfr(x),
                    #     axis=1)
                    # with tf.io.TFRecordWriter(tfr_name) as writer:
                    #     for serialized_item in tqdm(serialized_records.values):
                    #         writer.write(serialized_item.SerializeToString())
        #         dataset_raw = tf.data.TFRecordDataset(tfr_name)
        #         dataset = dataset_raw.map(self._deserialize_tokens)
        #         dataset_list.append(dataset)
        #
        # return dataset_list

    def read_tfr_dataset(self, tfr_file, nested_level=2):
        dataset_raw = tf.data.TFRecordDataset(tfr_file)
        dataset = dataset_raw.map(self._deserialize_tokens)
        return dataset

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
            return val1 if val2 is None or 0 < val2 < val1 else val1
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

    def _split_all(self, text):
        tokens = self._text_split(text, True, True)
        word_len = [len(w) for s in tokens for w in s]
        sent_len = [len(s) for s in tokens]
        flat_chars = [c for s in tokens for w in s for c in w]
        return flat_chars, word_len, sent_len

    def _flatten_to_lowest(self, nested_list):
        for elem in nested_list:
            if isinstance(elem, list):
                yield from self._flatten_to_lowest(elem)
            else:
                yield elem

    @staticmethod
    def _len_per_nested_list(token_list):
        """
        generator that returns length of each element in a nested list, recursively
        e.g. [[[1,2],[3]],[[4]]] would return [2,1] then [2,1,1]
        """
        while all(isinstance(e, list) for e in token_list):
            yield [len(e) for e in token_list]
            token_list = [elem for sublist in token_list for elem in sublist]

    def csv_to_pickle(self, split_characters=True, split_sentences=False, target_dir="data"):
        """
        pickle files when text is splitted, so the spacy text splitting only needs to be done once.
        A TF dataset can be created from pickle files
        """
        def b2c(flag):
            return 'Y' if flag else 'N'
        dataset_subdir = f"split_char_{b2c(split_characters)}_split_sent_{b2c(split_sentences)}"
        dataset_dir = os.path.join(target_dir, dataset_subdir)

        def mk_if_not_exist(d):
            if not os.path.isdir(d):
                os.mkdir(d)

        mk_if_not_exist(dataset_dir)
        for label in self.labels:
            mk_if_not_exist(os.path.join(dataset_dir, label))

        for df_chunk in self.text_df_gen:
            tqdm.pandas()
            df_chunk.progress_apply(lambda x: self._obs_to_pickle(
                self._text_split(x[self.text_col], split_characters=split_characters, split_sentences=split_sentences),
                x[self.label_col],
                x[self.id_col],
                dataset_dir
            ), axis=1)

        return dataset_dir

    @staticmethod
    def _obs_to_pickle(text_data, label, doc_id, dataset_dir):
        with open(os.path.join(dataset_dir, label,  doc_id + '.pkl'), 'wb') as f:
            pickle.dump(text_data, f)

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

    def _serialize_tokens_tfr(self, row):
        tokens = self._text_split(row[1][self.text_col], True, True)
        nested_len = [e for e in self._len_per_nested_list(tokens)]
        label = row[1][self.label_col]
        doc_id = row[0]

        context_features = tf.train.Features(feature={
            'doc_id': bytes_feature(doc_id),
            'label': bytes_feature(label)
        })

        feature_list = {
            'characters': tf.train.FeatureList(feature=_get_seq_feature(self._flatten_to_lowest(tokens)))
        }
        for nest_depth, len_list in enumerate(nested_len):
            feature_list['len_level_' + str(nest_depth)] = \
                tf.train.FeatureList(feature=_get_seq_feature(len_list, encoder=int64_feature))

        sequence_features = tf.train.FeatureLists(feature_list=feature_list)

        sequence_example = tf.train.SequenceExample(
            context=context_features,
            feature_lists=sequence_features,
        )

        return sequence_example

    @staticmethod
    def _deserialize_tokens(observation, nested_depth=2):
        context_features = {
            "doc_id": tf.io.FixedLenFeature([], dtype=tf.string),
            "label": tf.io.FixedLenFeature([], dtype=tf.string),
        }

        sequence_features = {
            "characters": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        }

        for i in range(nested_depth):
            sequence_features['len_level_' + str(i)] = tf.io.FixedLenSequenceFeature([], dtype=tf.int64)

        context, sequences = tf.io.parse_single_sequence_example(
            serialized=observation,
            context_features=context_features,
            sequence_features=sequence_features
        )
        res_dict = {
                   'doc_id': context['doc_id'],
                   'characters': sequences['characters'],
                   'len_level_0': sequences['len_level_0'],
                   'len_level_1': sequences['len_level_1']}

        for i in range(nested_depth):
            res_dict['len_level_' + str(i)] = sequences['len_level_' + str(i)]

        return res_dict, context['label']

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

        # todo convert dataset to RaggedTensors ?
        #   General approach for arbitrary split, and arbitrary sequence model in between.
