# Load submodules
import tensorflow as tf
import pandas as pd
import numpy as np
import regex as re
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import time
from tqdm import tqdm
import logging

LOGGER = logging.getLogger(__name__)


tf.compat.v1.disable_eager_execution()

# Load the dataset
LOGGER.debug(">> Loading the dataset ...")
lang_set = pd.read_csv('sentences.csv', sep='\t')
lang_set = lang_set.dropna()
LOGGER.debug(">> done.")

LOGGER.debug(">> Rearranging everything ...")
X, Y = [], []
for no, ln in enumerate(lang_set.cmn.unique()):
  langs = lang_set.loc[lang_set.cmn == ln]
  if langs.shape[0] < 500:
    continue
  langs = langs.iloc[:500, -1].tolist()
  X.extend(langs)
  Y.extend([ln] * len(langs))
LOGGER.debug(">> done.")

# Get the labels -> required for classification_report
labels = np.unique(Y, return_counts=True)[0]


def clean_text(string):
  res = re.sub(u'[0-9!@#$%^&*()_\-+{}|\~`\'";:?/.>,<]', ' ', string.lower(), flags=re.UNICODE)
  return re.sub(r'[ ]+', ' ', res.lower()).strip()


X = [clean_text(s) for s in X]

bow_chars = CountVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=700000).fit(X)
delattr(bow_chars, 'stop_words_')
target = LabelEncoder().fit_transform(Y)
features = bow_chars.transform(X)

train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.2)
del features


def convert_sparse_matrix_to_sparse_tensor(X, limit=5):
  coo = X.tocoo()
  indices = np.mat([coo.row, coo.col]).transpose()
  coo.data[coo.data > limit] = limit
  return tf.compat.v1.SparseTensorValue(indices, coo.col, coo.shape), tf.compat.v1.SparseTensorValue(indices, coo.data, coo.shape)


class NSECLanguageDetection(object):
  def __init__(self, learning_rate):
    super(NSECLanguageDetection, self).__init__()
    
    self.X = tf.compat.v1.sparse_placeholder(tf.int32)
    self.W = tf.compat.v1.sparse_placeholder(tf.int32)
    self.Y = tf.compat.v1.placeholder(tf.int32, [None])
    embeddings = tf.Variable(tf.compat.v1.truncated_normal([train_X.shape[1], 64]))
    embed = tf.nn.embedding_lookup_sparse(embeddings, self.X, self.W, combiner='mean')
    self.logits = tf.compat.v1.layers.dense(embed, len(labels))
    self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.logits,
      labels=self.Y)
    )
    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#
# Default TF (1.X) workflow
#
sess = tf.compat.v1.InteractiveSession()
model = NSECLanguageDetection(learning_rate=1e-4)
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 64
for e in range(5):
  lasttime = time.time()
  train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
  pbar = tqdm(
    range(0, train_X.shape[0], batch_size), desc='Train loop'
  )
  for i in pbar:
    batch_x = convert_sparse_matrix_to_sparse_tensor(train_X[i: min(i + batch_size, train_X.shape[0])])
    batch_y = train_Y[i: min(i + batch_size, train_X.shape[0])]
    acc, cost, _ = sess.run(
      [model.accuracy, model.cost, model.optimizer],
      feed_dict={
        model.X: batch_x[0],
        model.W: batch_x[1],
        model.Y: batch_y
      },
    )
    assert not np.isnan(cost)
    train_loss += cost
    train_acc += acc
    pbar.set_postfix(cost=cost, accuracy=acc)
  
  pbar = tqdm(range(0, test_X.shape[0], batch_size), desc='Test loop')
  for i in pbar:
    batch_x = convert_sparse_matrix_to_sparse_tensor(test_X[i: min(i + batch_size, test_X.shape[0])])
    batch_y = test_Y[i: min(i + batch_size, test_X.shape[0])]
    batch_x_expand = np.expand_dims(batch_x, axis=1)
    acc, cost = sess.run(
      [model.accuracy, model.cost],
      feed_dict={
        model.X: batch_x[0],
        model.W: batch_x[1],
        model.Y: batch_y
      },
    )
    test_loss += cost
    test_acc += acc
    pbar.set_postfix(cost=cost, accuracy=acc)
  
  train_loss /= train_X.shape[0] / batch_size
  train_acc /= train_X.shape[0] / batch_size
  test_loss /= test_X.shape[0] / batch_size
  test_acc /= test_X.shape[0] / batch_size
  
  LOGGER.debug('time taken:', time.time() - lasttime)
  LOGGER.debug(
    'epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'
    % (e, train_loss, train_acc, test_loss, test_acc)
  )

real_Y, predict_Y = [], []

pbar = tqdm(
  range(0, test_X.shape[0], batch_size), desc='Validation loop'
)
for i in pbar:
  batch_x = convert_sparse_matrix_to_sparse_tensor(test_X[i: min(i + batch_size, test_X.shape[0])])
  batch_y = test_Y[i: min(i + batch_size, test_X.shape[0])].tolist()
  predict_Y += np.argmax(
    sess.run(
      model.logits, feed_dict={model.X: batch_x[0], model.W: batch_x[1], model.Y: batch_y}
    ),
    1,
  ).tolist()
  real_Y += batch_y

print(
  metrics.classification_report(
    real_Y, predict_Y, target_names=labels
  )
)