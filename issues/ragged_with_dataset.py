import tensorflow as tf
import numpy as np

# see https://github.com/tensorflow/tensorflow/issues/24520 which is annoying when using Datasets & Keras fit
# see https://stackoverflow.com/questions/52582275/tf-data-with-multiple-inputs-outputs-in-keras

# at the moment it seems that the dataset tensors must be of fixed shape (and therefore padded) ?
# maybe unless using padded batch dataset

sent1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
sent2 = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
sent1 = np.reshape(sent1, (2, 4))
sent2 = np.reshape(sent2, (2, 4))

labels = np.array([1.0, 0.0, ], dtype=np.float32)
labels = np.reshape(labels, (2, 1))


def generator():
    for s1, s2, l in zip(sent1, sent2, labels):
        yield {"input_1": s1, "input_2": s2}, l


dataset = tf.data.Dataset.from_generator(generator,
                                         output_types=({"input_1": tf.int32, "input_2": tf.int32}, tf.float32),
                                         output_shapes=({"input_1": [4, ], "input_2": [4, ]}, [1, ])
                                         )
dataset = dataset.batch(1)

for item in dataset:
    print(item)

token_in = tf.keras.layers.Input(shape=(None,), name='input_1', dtype=tf.int32, ragged=False)
row_len_in = tf.keras.layers.Input(shape=(None,), name='input_2', dtype=tf.int32)

ragged_in = tf.RaggedTensor.from_row_lengths(token_in, [2, 2], validate=False)

embedded_1 = tf.keras.layers.Embedding(9, 4)(ragged_in)
prediction = tf.keras.layers.LSTM(2)(embedded_1)

model = tf.keras.Model(inputs=[token_in, row_len_in], outputs=prediction)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(dataset, steps_per_epoch=1)

# todo: check https://github.com/tensorflow/text/issues/174 to use Dataset with Ragged Tensors
#   try different (custom) tokenizers, split by '.', by character, etc..
# todo: check custom training as alternative approach if dataset/keras doesn't work
#  https://www.tensorflow.org/tutorials/customization/custom_training

