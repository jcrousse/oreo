"""
applying a model on a specific dimentions of a ragged tensor
"""

import tensorflow as tf

ragged_simple_seq = tf.ragged.constant([
    ['First', 'observation', 'single', 'sentence'],
    ['Second', 'observation', 'another', 'sentence']])


# Look up the embedding for each word.
word_buckets = tf.strings.to_hash_bucket_fast(ragged_simple_seq, 5)
embedded = tf.keras.layers.Embedding(5, 2)(word_buckets)
sentence_encoding = tf.keras.layers.LSTM(4)(embedded)
print(sentence_encoding)


ragged_nested_seq = tf.ragged.constant([
    [['First_observation', 'sentence1'],
     ['First_observation', 'sentence2']],
    [
        ['Second_observation', 'another_sentence']
    ]])

# ragged_nested_seq = tf.ragged.constant([
#     [[['o1_s1_w1_c1', 'o1_s1_w1_c2'], ['o1_s1_w2_c1']],
#      [['o1_s2_w1_c1'], ['o1_s2_w2_c1']]],
#     [
#         [['o2_s1_w1_c1'], ['o2_s1_w2_c1']]
#     ]])

word_buckets_nested = tf.strings.to_hash_bucket_fast(ragged_nested_seq, 5)
embedded_nested = tf.keras.layers.Embedding(5, 4)(word_buckets_nested)

print(embedded_nested)
print(embedded_nested.bounding_shape())
print(embedded_nested.flat_values)
reshaped_ragged = tf.RaggedTensor.from_row_lengths(
    values=embedded_nested.flat_values,
    row_lengths=[2, 2, 2]  # words per sentence
)
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
