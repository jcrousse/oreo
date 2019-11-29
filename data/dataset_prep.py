"""
reads the IMDB data from a CSV and generates a TF dataset.
Tokenization handled by SpaCy.

"""

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
# import tensorflow as tf

from data.data_config import IMDB_CSV, LABELS, LABEL_COL, ID_COL, TEXT_COL, ENCODING


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


class TextDataSetPrep:
    def __init__(self,
                 csv_path=IMDB_CSV,
                 labels=LABELS,
                 label_col=LABEL_COL,
                 id_col=ID_COL,
                 text_col=TEXT_COL,
                 encoding=ENCODING,
                 spacy_model='en_core_web_sm',
                 nrows=None,
                 ):
        self.csv_path = csv_path
        self.labels = labels
        self.label_col = label_col
        self.text_col = text_col
        self.id_col = id_col
        self.columns = [self.id_col, self.label_col]
        full_df = pd.read_csv(self.csv_path, encoding=encoding, nrows=nrows, usecols=self.columns)
        self.text_df = full_df.loc[full_df[label_col].isin(self.labels), self.columns]

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

    def get_tf_dataset(self,
                       batch_size=256,
                       n_per_label=None,
                       use_tf_record=True,
                       dataset_split=None,
                       seed=None):

        selected_ids = self._selected_ids(n_per_label=n_per_label, seed=seed)
        dataset_ids = self._get_ids_per_dataset(selected_ids, dataset_split)
        dataset_list = []
        for id_list in dataset_ids:
            pass
            # dataset = tf.data.experimental.CsvDataset(self.csv_path,
            #                                           select_cols=self.tf_col_indices,
            #                                           record_defaults=["", "", ""])
            # def filter_fn(x, y, z):
            #     return True if y == 'label' else False
            #
            # def map_fn(x, y, z):
            #     return (x, self._text_split(y)), z
            # dataset = dataset.map(map_fn)
            # dataset = dataset.filter(filter_fn)
            # dataset_list.append(dataset)
            # _ = 1
        return dataset_list

    def _selected_ids(self, n_per_label=1000, seed=None):
        def min_row(val1, val2):
            return val2 if val2 is None or 0 < val2 < val1 else val1
        return self.text_df.groupby(self.label_col, group_keys=False)\
                   .apply(lambda x: x.sample(min_row(len(x), n_per_label), random_state=seed)).loc[:, self.id_col].values

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
