import tensorflow as tf

vocab_size = 20
text = [
    "this is a positive example".split(' '),
    "this one is definitely negative and has more words".split(' ')
]

words_lengths = [[len(w) for w in sent] for sent in text]
sentence_lengths = [len(s) for s in text]

words_lengths_t = tf.ragged.constant(words_lengths)
sentence_lengths_t = tf.constant(sentence_lengths)
flat_characters = [[c for w in sent for c in w] for sent in text]

text_t = tf.ragged.constant(flat_characters)
int_chars = tf.dtypes.cast(tf.strings.to_hash_bucket_fast(text_t, vocab_size), tf.float32)


embedded = tf.keras.layers.Embedding(vocab_size, 2)(int_chars)

# w_lens_flat = tf.reshape(words_lengths_t, [-1])  # not necessary step in eager mode, but needed in graph

w_lens_flat = words_lengths_t.flat_values
embedd_flat = embedded.flat_values

lstm_lvl1_in = tf.keras.layers.Lambda(
    lambda x: tf.RaggedTensor.from_row_lengths(x, w_lens_flat)
)(embedd_flat)


lstm_lvl1_out = tf.keras.layers.LSTM(4)(lstm_lvl1_in)

print(lstm_lvl1_out)

lstm_lvl2_in = tf.keras.layers.Lambda(
    lambda x: tf.RaggedTensor.from_row_lengths(x, sentence_lengths_t)
)(lstm_lvl1_out)

lstm_lvl2_out = tf.keras.layers.LSTM(4)(lstm_lvl2_in)

print(lstm_lvl2_out)


# extra steps to take in graph mode:
#   -ragged in to ensure graph can compile with input shapes
#   -reshape inside the graph to go back to flat vals