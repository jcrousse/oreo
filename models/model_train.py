import tensorflow as tf
from data.dataset_prep import TextDataSetPrep
from models.basic_sequence import model
from models.config import vocab_size

tds = TextDataSetPrep(nrows=200)  # need pos and neg examples
x, y = tds.get_ragged_tensors_dataset()

word_buckets = tf.strings.to_hash_bucket_fast(x, vocab_size).to_tensor()
evaluation = tf.keras.metrics.CategoricalAccuracy()


model.summary()

model.fit(word_buckets, y, batch_size=512, epochs=40)

print(f"Categorical accuracy after train: {evaluation(model(word_buckets), y).numpy()}")

model.evaluate(word_buckets, y)
