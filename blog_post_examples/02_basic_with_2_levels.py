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

y = tf.constant([0.0, 1.0])

char_tokens = tf.keras.layers.Input(shape=(None,), ragged=True, name="int_chars")
w_lens = tf.keras.layers.Input(shape=(None,), ragged=True, name="w_len", dtype=tf.int32)
s_len = tf.keras.layers.Input(shape=(None, ), name="s_len", dtype=tf.int32)

embedd_size = 8
embedded = tf.keras.layers.Embedding(vocab_size, embedd_size)(char_tokens)

w_lens_flat = w_lens.flat_values
embedd_flat = embedded.flat_values

seq_lvl1_in = tf.keras.layers.Lambda(
    lambda x: tf.RaggedTensor.from_row_lengths(x, w_lens_flat)
)(embedd_flat)

seq_lvl1_out = tf.keras.layers.LSTM(4)(seq_lvl1_in)


classifier = tf.keras.layers.Dense(1, activation='softmax', name='classifier')(seq_lvl1_out)
model = tf.keras.Model(inputs=[s_len, char_tokens, w_lens], outputs=classifier)

model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'],
)


model.fit({
    'int_chars': int_chars,
    'w_len': words_lengths_t,
    's_len': sentence_lengths_t},
    y, epochs=2)

#todo: Then the two-levels thing
# experimental_run_tf_function=False

# question posted on https://stackoverflow.com/questions/60692511/converting-eager-mode-code-to-graph-mode-with-ragged-tensors