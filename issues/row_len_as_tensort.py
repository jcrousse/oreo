import tensorflow as tf
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Working case of graph mode + reshape to ragged + usage of dataset
# todo: Use Keras Sequential with custom layer as advised in
#   https://www.tensorflow.org/guide/migrate#use_keras_training_loops


def name_cols(ft, rl):
    return {"flat_tokens": ft, "row_lengths": rl}


def input_fn():
    flat_tokens = np.reshape(
        np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        (2, 4))
    row_lengths = [
        [4],
        [4]
    ]

    labels = np.reshape(
        np.array([1.0, 0.0, ], dtype=np.float32),
        (2, 1))

    def id_to_obs(idx):
        return flat_tokens[idx.numpy()], row_lengths[idx.numpy()]

    dataset = tf.data.Dataset.from_tensor_slices([0, 1])
    dataset = dataset.map(lambda x: tf.py_function(
        func=id_to_obs, inp=[x], Tout=(tf.int32, tf.int32)))
    dataset = dataset.map(name_cols)
    dataset = dataset.padded_batch(1, {"flat_tokens": [4, ], "row_lengths": [1, ]})
    dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(labels)))
    return dataset


token_in = tf.keras.layers.Input(shape=(None,), name='flat_tokens', dtype=tf.int32)
row_len_in = tf.keras.layers.Input(shape=(None,), name='row_lengths', dtype=tf.int32)

flat_in = tf.reshape(token_in, [-1])
flat_rl = tf.reshape(row_len_in, [-1])

ragged_in = tf.keras.layers.Lambda(
        lambda x: tf.RaggedTensor.from_row_lengths(flat_in, flat_rl)
    )(flat_in)

embedded = tf.keras.layers.Embedding(10, 8)(ragged_in)

prediction = tf.keras.layers.LSTM(2)(embedded)

model = tf.keras.Model(inputs=[token_in, row_len_in], outputs=prediction)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(input_fn(),  steps_per_epoch=1)
