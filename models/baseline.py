import tensorflow as tf
from data.dataset_prep import TextDataSetPrep

tds = TextDataSetPrep(nrows=20000)  # need pos and neg examples
x, y = tds.get_ragged_tensors_dataset()

vocab_size = 10000
word_buckets = tf.strings.to_hash_bucket_fast(x, vocab_size).to_tensor()
evaluation = tf.keras.metrics.CategoricalAccuracy()

inputs = tf.keras.layers.Input(shape=(None, ), ragged=False)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))


model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'])
model.summary()

print(f"Categorical accuracy before train: {evaluation(model(word_buckets), y).numpy()}")
model.fit(word_buckets, y, batch_size=512, epochs=40)

print(f"Categorical accuracy after train: {evaluation(model(word_buckets), y).numpy()}")

model.evaluate(word_buckets, y)