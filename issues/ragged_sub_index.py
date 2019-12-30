"""
applying a model on a specific dimentions of a ragged tensor
"""

import tensorflow as tf
from data.dataset_prep import TextDataSetPrep

VOCAB_N_L0 = 5
EMBEDDING_SIZE_L0 = 2
ENCODING_SIZE_L1 = 4


ragged_simple_seq = tf.ragged.constant([
    ['First', 'observation', 'single', 'sentence'],
    ['Second', 'observation', 'another', 'sentence']])

ds = TextDataSetPrep(nrows=1).read_tfr_dataset("train_small.tfr")

ts_list = []
len_l0 = []
len_l1 = []
for item in ds.take(2):
    ts_list.append(item[0]['characters'])
    len_l0.append(item[0]['len_level_0'])
    len_l1.append(item[0]['len_level_1'])



input_tokens = tf.keras.layers.Input(shape=(None,), name='characters', dtype=tf.string)
input_grouping = tf.keras.layers.Input(shape=(None,), name='len_level_0', dtype=tf.int32)

word_buckets = tf.strings.to_hash_bucket_fast(input_tokens, VOCAB_N_L0)
embedded = tf.keras.layers.Embedding(VOCAB_N_L0, EMBEDDING_SIZE_L0)(word_buckets)

reshaped = tf.RaggedTensor.from_row_lengths(
    values=embedded,
    row_lengths=tf.reshape(input_grouping, [-1]))
# yet another situation where the combination of Keras and RaggedTensor sucks.
# todo next: reshape without ragged (use padding), and try to compile,
#  then try to print out the tf.reshape(input_grouping, [-1] to check if it is the value we want.
#  if hardcoding the same value as the reshaped tensor works, then we can raise an issue. 
encoded = tf.keras.layers.LSTM(ENCODING_SIZE_L1)(reshaped)

model = tf.keras.Model(inputs=[input_tokens, input_grouping], outputs=encoded)
model({'characters': ts_list[0], 'len_level_0': len_l0[0]})
model.predict(ds.take(2))

ragged_by_char = tf.ragged.constant(ts_list)
word_buckets = tf.strings.to_hash_bucket_fast(ragged_by_char, VOCAB_N_L0)
embedded = tf.keras.layers.Embedding(VOCAB_N_L0, EMBEDDING_SIZE_L0)(word_buckets)

regrouped = tf.RaggedTensor.from_row_lengths(
    values=embedded,
    row_lengths=len_l0  # words per sentence
)

sentence_encoding = tf.keras.layers.LSTM(ENCODING_SIZE_L1)(embedded)







print(reshaped_ragged)
sentence_encoding_nested = tf.keras.layers.LSTM(4)(reshaped_ragged)
print(sentence_encoding_nested)
reshaped_by_obs = tf.RaggedTensor.from_row_lengths(
    values=sentence_encoding_nested,
    row_lengths=[2, 1]  # words per sentence
)
print(reshaped_by_obs)
level_2_seq = tf.keras.layers.LSTM(4)(reshaped_by_obs)

print(level_2_seq)
# todo: make the above for arbitrary N nested levels and user selected sequence model