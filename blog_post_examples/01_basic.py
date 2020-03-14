import tensorflow as tf

vocab_size = 20
text = [
    "this is a positive example".split(' '),
    "this one is definitely negative and has more words".split(' ')
]


inputs = tf.keras.layers.Input(shape=(None, ), ragged=True)
embedded = tf.keras.layers.Embedding(vocab_size, 8)(inputs)
lstm_out = tf.keras.layers.LSTM(16)(embedded)
classifier = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(lstm_out)
model = tf.keras.Model(inputs=inputs, outputs=classifier)

model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'])


ragged_text = tf.ragged.constant(text)
x = tf.dtypes.cast(tf.strings.to_hash_bucket_fast(ragged_text, vocab_size), tf.float32)


y = tf.constant([[0.0, 1.0], [0.0, 1.0]])
model.fit(x, y, epochs=2)
