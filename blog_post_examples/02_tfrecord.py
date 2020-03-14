import tensorflow as tf

vocab_size = 20
text = [
    "this is a positive example".split(' '),
    "this one is definitely negative and has more words".split(' ')
]

labels = [1.0, 0.0]

words_lengths = [[len(w) for w in sent] for sent in text]
sentence_lengths = [len(s) for s in words_lengths]
flat_characters = [[c for w in sent for c in w] for sent in text]

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def get_seq_feature(tokens, encoder=bytes_feature):
    tokens_features = []
    for elem in tokens:
        tokens_features.append(encoder(elem))
    return tokens_features


for idx, item_data in enumerate(zip(flat_characters, words_lengths, sentence_lengths, labels)):
    obs_chars, words_ln, sent_ln, obs_label = item_data
    context_features = tf.train.Features(feature={
        'label': _float_feature(obs_label),
        'sent_len': int64_feature(sent_ln)
    })

    feature_list = {
        'characters': tf.train.FeatureList(feature=get_seq_feature(obs_chars)),
        'words_len': tf.train.FeatureList(feature=get_seq_feature(words_ln, int64_feature)),

    }

    sequence_features = tf.train.FeatureLists(feature_list=feature_list)

    sequence_example = tf.train.SequenceExample(
        context=context_features,
        feature_lists=sequence_features,
    )

    with tf.io.TFRecordWriter(f"example_{idx}.tfr") as writer:
        writer.write(sequence_example.SerializeToString())


def deserialize_tokens(observation):

    context_features = {
        "label": tf.io.FixedLenFeature([], dtype=tf.float32),
        'sent_len': tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    sequence_features = {
        "characters": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'words_len':  tf.io.FixedLenSequenceFeature([], dtype=tf.int64),


    }

    context, sequences = tf.io.parse_single_sequence_example(
        serialized=observation,
        context_features=context_features,
        sequence_features=sequence_features
    )
    res_dict = {
        'characters': sequences['characters'],
        'words_len': sequences['words_len'],
        'sent_len': context['sent_len']
    }

    return res_dict, context['label']

# todo: feed ragged to LSTM, then second level LSTM to output?


dataset_raw = tf.data.TFRecordDataset(["example_0.tfr", "example_1.tfr"])
dataset = dataset_raw.map(deserialize_tokens)

input_chars = tf.keras.layers.Input(shape=(None,), name='characters', dtype=tf.string)
input_lens = tf.keras.layers.Input(shape=(None,), name='words_len', dtype=tf.int32)
input_sent_len = tf.keras.layers.Input(shape=(None,), name='sent_len', dtype=tf.int32)

char_ints = tf.strings.to_hash_bucket_fast(input_chars, vocab_size)

reshaped_len = tf.reshape(input_lens, [-1])
flat_chars = tf.reshape(char_ints, [-1])

ragged_in = tf.keras.layers.Lambda(
    lambda x: tf.RaggedTensor.from_row_lengths(x, reshaped_len)
)(flat_chars)

embedded = tf.keras.layers.Embedding(vocab_size, 8)(ragged_in)

lstm_out = tf.keras.layers.LSTM(1)(embedded)

# ragged_level2 = tf.keras.layers.Lambda(
#     lambda x: tf.RaggedTensor.from_row_lengths(lstm_out, input_sent_len)
# )(flat_chars)

model = tf.keras.Model(inputs=[input_chars, input_lens], outputs=lstm_out)

model.compile(
    tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc'],
    experimental_run_tf_function=False)

model.fit(dataset, steps_per_epoch=1)

# https://github.com/tensorflow/tensorflow/issues/33729