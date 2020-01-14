import tensorflow as tf
import numpy as np

# working example where the model input type is already a ragged tensor

input_sequence = tf.ragged.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
print(input_sequence.bounding_shape())
labels = np.reshape(
    np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    (2, 2))

sequence_in = tf.keras.layers.Input(shape=(None,), name='input_1', dtype=tf.int32, ragged=True)

# embedded_ragged = tf.keras.layers.Embedding(9, 4)(ragged_in)
embedded = tf.keras.layers.Embedding(9, 4)(sequence_in)
lstm_out = tf.keras.layers.LSTM(64)(embedded)
prediction = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(lstm_out)


model = tf.keras.Model(inputs=sequence_in, outputs=prediction)
model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'])

model.fit(input_sequence, labels, steps_per_epoch=1)
