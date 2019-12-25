import tensorflow as tf
from data.dataset_prep import TextDataSetPrep

tds = TextDataSetPrep()

dataset = tds.read_tfr_dataset('train_small.tfr')

char_vocab_size = 200
word_vector_len = 50

def tfr_to_ragged(example, label):
    nested_0 = tf.RaggedTensor.from_row_lengths(
            values=example['characters'],
            row_lengths=example['len_level_1']
        )
    nested_1 = tf.RaggedTensor.from_row_lengths(
        values=nested_0,
        row_lengths=example['len_level_0']
    )
    return nested_0, label

dataset_ragged = dataset.map(tfr_to_ragged)
dataset = dataset_ragged.batch(32).prefetch(256)

# x = tf.keras.layers.Input(shape=(None, None, ), ragged=True, name='input_x')
# y = tf.keras.layers.Input(shape=(None,), ragged=False, name='input_y')


for x, y in dataset:
    x_cropped = x
    word_buckets = tf.strings.to_hash_bucket_fast(x_cropped, char_vocab_size)
    embedded = tf.keras.layers.Embedding(char_vocab_size, 8)(word_buckets)
    # need to turn each word into an obwervation to apply LSTM
    tf.ragged.stack()
    word_embeddings = tf.keras.layers.LSTM(word_vector_len)(embedded)
    _ = 1
# flat_characters = inputs['characters']
# words_len = inputs['len_level_1']
# sent_len = inputs['len_level_0']


# model = tf.keras.Model(inputs=[x, y], outputs=embedded)


# model.compile(
#     tf.keras.optimizers.Adam(),
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     metrics=['acc'])

# model.fit(dataset)
