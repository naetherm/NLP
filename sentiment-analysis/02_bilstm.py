import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

BUFFER_SIZE = 10000
BATCH_SIZE = 64
HIDDEN_SIZE = 64
LR = 1e-4


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_' + metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_' + metric])
  plt.show()


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)

test_dataset = test_dataset.padded_batch(BATCH_SIZE)

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(encoder.vocab_size, HIDDEN_SIZE),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HIDDEN_SIZE)),
  tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(LR),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

