import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# question posted at
# https://stackoverflow.com/questions/59632120/converting-tensor-to-ragged-tensor-in-graph-mode-using-keras


def input_fn():
    input_sequence = np.reshape(
        np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        (2, 4))

    labels = np.reshape(
        np.array([1.0, 0.0, ], dtype=np.float32),
        (2, 1))

    dataset = tf.data.Dataset.from_tensor_slices((input_sequence, labels)).batch(1)

    return dataset


sequence_in = tf.keras.layers.Input(shape=(4,), name='input_sequence', dtype=tf.int32, ragged=False)

ragged_in = tf.keras.layers.Lambda(
    lambda x: tf.RaggedTensor.from_row_lengths(x, [2, 2])
)(sequence_in)

embedded_ragged = tf.keras.layers.Embedding(9, 4)(ragged_in)

embedded_tensor = tf.keras.layers.Embedding(9, 4)(sequence_in)

flat_tensor = tf.reshape(embedded_tensor, [-1, 16])
prediction = tf.keras.layers.Dense(2)(flat_tensor)

model = tf.keras.Model(inputs=sequence_in, outputs=prediction)
model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'])

model.fit(input_fn(), steps_per_epoch=1)
