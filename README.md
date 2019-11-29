# Work In Progress....
Playing with TF2.0 text and datasets 

need at least tensorflow nightly 20191111 for ragged tensor support with Keras layers.
see https://github.com/tensorflow/tensorflow/issues/27170

__TODO NEXT__: Generate a TFRecord of examples (in batches) rather than using CsvDataset

```python
import tensorflow as tf

val_ragged = tf.ragged.constant([[1, 2, 3], [1, 2], [1, 2, 3, 4]])

val_tensor = val_ragged.to_tensor()

inputs = tf.keras.layers.Input(shape=(None, None,), ragged=False)
outputs = tf.keras.layers.Embedding(5, 4)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# this model with normal tensor works
print(model(val_tensor))

inputs_ragged = tf.keras.layers.Input(shape=(None, None,), ragged=True)
outputs_ragged = tf.keras.layers.Embedding(5, 4)(inputs_ragged)
model_ragged = tf.keras.Model(inputs=inputs_ragged, outputs=outputs_ragged)

# this one with RaggedTensor doesn't
print(model_ragged(val_ragged))

```