import tensorflow as tf
# read a dataset with a flat tensor of tokens, then convert them to a raggedtensor
# using a function such as from_row_length (but we could consider using any from the doc)

# our example dataset has two observations, one with two sequences of length 2 each, and one with one sequence of len 2
flat_tokens = [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]
row_lengths = [
    [2, 2],
    [2, 2]
]

labels = [0.0, 1.0]

d = tf.data.Dataset.from_tensor_slices([0, 1])

# transform a string tensor to upper case string using a Python function
def upper_case_fn(t):
    return flat_tokens[t.numpy()]

d = d.map(lambda x: tf.py_function(func=upper_case_fn,
                                   inp=[x], Tout=tf.int64))  # ==> [ "HELLO", "WORLD" ]

for item in d:
    print(item)


def id_to_obs(idx):
    return flat_tokens[idx.numpy()], row_lengths[idx.numpy()]


def name_cols(ft, rl):
    return {"flat_tokens": ft, "row_lengths": rl}

dataset = tf.data.Dataset.from_tensor_slices([0, 1])
dataset = dataset.map(lambda x: tf.py_function(
    func=id_to_obs, inp=[x], Tout=(tf.int32, tf.int32)))
dataset = dataset.map(name_cols)
dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(labels)))

for item in dataset:
    print(item)

token_in = tf.keras.layers.Input(shape=(4,), name='flat_tokens', dtype=tf.int32)
row_len_in = tf.keras.layers.Input(shape=(2,), name='row_lengths', dtype=tf.int32)
embedded = tf.keras.layers.Embedding(6, 4)(token_in)
prediction = tf.keras.layers.LSTM(1)(embedded)

model = tf.keras.Model(inputs=[token_in, row_len_in], outputs=prediction)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(dataset)

# subject to issue https://github.com/tensorflow/tensorflow/issues/24520