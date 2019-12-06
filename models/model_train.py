import tensorflow as tf
from data.dataset_prep import TextDataSetPrep
from models.basic_sequence import model
from models.config import vocab_size

tds = TextDataSetPrep(nrows=20000)  # need pos and neg examples
MAX_SENT_LEN = 30
MAX_WORD_LEN = 15
MAX_REVIEW_LEN = 200


x, y = tds.get_ragged_tensors_dataset()

x_cropped = x[:, :MAX_REVIEW_LEN]

word_buckets = tf.dtypes.cast(tf.strings.to_hash_bucket_fast(x_cropped, vocab_size), tf.float32)
evaluation = tf.keras.metrics.CategoricalAccuracy()


model.summary()

model.fit(word_buckets, y, batch_size=512, epochs=40)

print(f"Categorical accuracy after train: {evaluation(model(word_buckets), y).numpy()}")

model.evaluate(word_buckets, y)
