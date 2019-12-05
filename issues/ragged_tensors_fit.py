import tensorflow as tf

val_ragged = tf.ragged.constant([[1.0, 2.0, 3.0], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
y = tf.constant([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
val_tensor = val_ragged.to_tensor()

inputs = tf.keras.layers.Input(shape=(None, ), ragged=False)
embeddings = tf.keras.layers.Embedding(5, 4)(inputs)
lstm_out = tf.keras.layers.LSTM(8)(embeddings)
classifier = tf.keras.layers.Dense(
        2, activation='softmax', name='classifier')(lstm_out)
model = tf.keras.Model(inputs=inputs, outputs=classifier)
model.compile(
    tf.keras.optimizers.Adam(0.01),
    loss=tf.keras.losses.CategoricalCrossentropy())
_ = model(val_tensor)
model.fit(val_tensor, y, epochs=1)


inputs = tf.keras.layers.Input(shape=(None, ), ragged=True)
embeddings = tf.keras.layers.Embedding(5, 4)(inputs)
lstm_out = tf.keras.layers.LSTM(8)(embeddings)
classifier = tf.keras.layers.Dense(
        2, activation='softmax', name='classifier')(lstm_out)
model = tf.keras.Model(inputs=inputs, outputs=classifier)
model.compile(
    tf.keras.optimizers.Adam(0.01),
    loss=tf.keras.losses.CategoricalCrossentropy())
_ = model(val_ragged)
model.fit(val_ragged, y, epochs=1)

# Solution: Convert input ragged to float
