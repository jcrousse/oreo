import tensorflow as tf

vocab_size = 20
text = [
    "this is a positive example".split(' '),
    "this one is definitely negative and has more words".split(' ')
]

words_lengths = [[len(w) for w in sent] for sent in text]
sentence_lengths = [len(s) for s in words_lengths]
flat_characters = [[c for w in sent for c in w] for sent in text]

tokens = tf.keras.layers.Input(shape=(None, ), ragged=True)

embedded = tf.keras.layers.Embedding(vocab_size, 8)(tokens)
lstm_out = tf.keras.layers.LSTM(16)(embedded)
classifier = tf.keras.layers.Dense(1, activation='softmax', name='classifier')(lstm_out)
model = tf.keras.Model(inputs=tokens, outputs=classifier)

model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'])


ragged_text = tf.ragged.constant(text)
int_tokens = tf.dtypes.cast(tf.strings.to_hash_bucket_fast(ragged_text, vocab_size), tf.float32)


y = tf.constant([0.0, 1.0])
model.fit(int_tokens, y, epochs=2)
