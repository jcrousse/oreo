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
int_chars = tf.dtypes.cast(tf.strings.to_hash_bucket_fast(text_t, vocab_size), tf.float32)

char_tokens = tf.keras.layers.Input(shape=(None,), name="int_chars")
w_lens = tf.keras.layers.Input(shape=(None,), name="w_len", dtype=tf.int32)
s_len = tf.keras.layers.Input(shape=(None, ), name="s_len", dtype=tf.int32)


embedd_size = 8
embedded = tf.keras.layers.Embedding(vocab_size, embedd_size)(char_tokens)
# lstm_out = tf.keras.layers.LSTM(16)(embedded)

w_lens_flat = tf.reshape(w_lens, [-1])
embedd_flat = tf.reshape(embedded, [-1, embedd_size])

seq_lvl1_in = tf.keras.layers.Lambda(
    lambda x: tf.RaggedTensor.from_row_lengths(x, w_lens_flat)
)(embedd_flat)

seq_lvl1_out = tf.keras.layers.LSTM(4)(seq_lvl1_in)

# lstm_lvl2_in = tf.keras.layers.Lambda(
#     lambda x: tf.RaggedTensor.from_row_lengths(x, s_len)
# )(seq_lvl1_out)
#
# lstm_lvl2_out = tf.keras.layers.LSTM(4)(lstm_lvl2_in)

classifier = tf.keras.layers.Dense(1, activation='softmax', name='classifier')(seq_lvl1_out)
model = tf.keras.Model(inputs=[s_len, char_tokens, w_lens], outputs=classifier)

model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'])

# experimental_run_tf_function=False

ragged_text = tf.ragged.constant(flat_characters)
int_tokens = tf.dtypes.cast(tf.strings.to_hash_bucket_fast(ragged_text, vocab_size), tf.float32)


y = tf.constant([0.0, 1.0])
model.fit({
    'int_chars': int_chars,
    'w_len': words_lengths_t,
    's_len': sentence_lengths_t},
    y, epochs=2)

#todo: Then the two-levels thing
