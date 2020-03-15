import tensorflow as tf

vocab_size = 20
text = [
    "this is a positive example".split(' '),
    "this one is definitely negative and has more words".split(' ')
]

words_lengths = [len(w) for sent in text for w in sent]
sentence_lengths = [len(s) for s in text]

words_lengths_t = tf.constant(words_lengths)
sentence_lengths_t = tf.constant(sentence_lengths)
flat_characters = [c for sent in text for w in sent for c in w]

text_t = tf.constant(flat_characters)

int_tokens = tf.dtypes.cast(tf.strings.to_hash_bucket_fast(text_t, vocab_size), tf.float32)


embedded = tf.keras.layers.Embedding(vocab_size, 2)(int_tokens)

w_lens_flat = tf.reshape(words_lengths_t, [-1])  # not necessary step in eager mode, but needed in graph

lstm_lvl1_in = tf.keras.layers.Lambda(
    lambda x: tf.RaggedTensor.from_row_lengths(x, w_lens_flat)
)(embedded)


lstm_lvl1_out = tf.keras.layers.LSTM(4)(lstm_lvl1_in)

print(lstm_lvl1_out)

lstm_lvl2_in = tf.keras.layers.Lambda(
    lambda x: tf.RaggedTensor.from_row_lengths(x, sentence_lengths_t)
)(lstm_lvl1_out)

lstm_lvl2_out = tf.keras.layers.LSTM(4)(lstm_lvl2_in)

print(lstm_lvl2_out)
