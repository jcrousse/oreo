import tensorflow as tf
from data.dataset_prep import TextDataSetPrep


tds = TextDataSetPrep(nrows=20000)  # need pos and neg examples
MAX_SENT_LEN = 30
MAX_WORD_LEN = 15    # 15
MAX_REVIEW_LEN = 200  # 200
VOCAB_SIZE = 200  # not too many possible characters (but bear in mind accents, upper/lower,...
CHAR_EMBEDD_DIM = 8  # 3
WORD_VECTORS_LEN = 128  # 256
REVIEW_EMBEDDING = 64

x, y = tds.get_ragged_tensors_dataset(split_characters=True)

x_cropped = x[:, :MAX_REVIEW_LEN, :MAX_WORD_LEN]

word_buckets = tf.strings.to_hash_bucket_fast(x_cropped, VOCAB_SIZE).to_tensor()
evaluation = tf.keras.metrics.CategoricalAccuracy()

inputs = tf.keras.layers.Input(shape=(None, None, ), ragged=False)

embedded_c = tf.keras.layers.Embedding(VOCAB_SIZE, CHAR_EMBEDD_DIM)(inputs)
reshaped = tf.reshape(embedded_c, (-1, MAX_WORD_LEN, CHAR_EMBEDD_DIM))
embedded_w = tf.keras.layers.LSTM(WORD_VECTORS_LEN)(reshaped)
group_by_review = tf.reshape(embedded_w, (-1, MAX_REVIEW_LEN, WORD_VECTORS_LEN))
lstm_review = tf.keras.layers.LSTM(REVIEW_EMBEDDING)(group_by_review)
dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense1')(lstm_review)
classifier = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(dense1)
model = tf.keras.Model(inputs=inputs, outputs=classifier)

model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'])

res = model(word_buckets)

model.summary()

model.fit(word_buckets, y, batch_size=512, epochs=100)

print(f"Categorical accuracy after train: {evaluation(model(word_buckets), y).numpy()}")

model.evaluate(word_buckets, y)
