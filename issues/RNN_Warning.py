import tensorflow as tf
import numpy as np


from tensorflow.python.framework.ops import disable_eager_execution
# COMMENT THE LINE BELOW TO REMOVE THE WARNING
disable_eager_execution()


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
embedded_tensor = tf.keras.layers.Embedding(10, 8)(sequence_in)
prediction = tf.keras.layers.LSTM(2)(embedded_tensor)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10, 8),
    tf.keras.layers.LSTM(2)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(input_fn(),  steps_per_epoch=1)
