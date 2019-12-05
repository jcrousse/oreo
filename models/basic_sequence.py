import tensorflow as tf
from models.config import vocab_size


inputs = tf.keras.layers.Input(shape=(None, ), ragged=False)

embedded = tf.keras.layers.Embedding(vocab_size, 64)(inputs)
lstm_out = tf.keras.layers.LSTM(64)(embedded)
classifier = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(lstm_out)
model = tf.keras.Model(inputs=inputs, outputs=classifier)

model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'])
