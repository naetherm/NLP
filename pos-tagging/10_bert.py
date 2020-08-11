
# If using the cased variant set those variables to ^cased too
BERT_VOCAB = 'uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = 'uncased_L-12_H-768_A-12/bert_config.json'
USE_LOWER = True

num_epochs = 5
batch_size = 8
warmup_proportion = 0.1
# Learning Rate
LR = 2e-5

seq_len = 50

import os
import sys
import time
import regex as re

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

#
# The parsing from the given training and test set files.
# If you want to change the task from Entity- to POS-Tagging just exchange the
# marked line
#


# Read and parse train and test sets
def parse(file):
    with open(file, 'r', encoding='utf-8') as fin:
        texts = fin.readlines()
    x, y = [], []
    for t in texts:
        # Skip uninformative lines
        if "-DOCSTART-" in t or len(t)<=1:
            continue
        split_ = t.split()
        x.append(split_[0])
        y.append(split_[1])
    return x, y


x_train, y_train = parse('eng.train')
x_test, y_test = parse('eng.testa')

#
# Just some length padding and extraction of all tags
#


def iter_seq(X):
    return np.array([X[i:i+seq_len] for i in range(0, len(X)-seq_len, 1)])


def to_train_seq(*args):
  return  [iter_seq(x) for x in args]


x_train, y_train = to_train_seq(x_train, y_train)
x_test, y_test = to_train_seq(x_test, y_test)

tag2idx = { "PAD": 0 }

for no, u in enumerate(np.unique(y_train)):
    tag2idx[u] = no + 1
print("generated tagset: {}".format(tag2idx))

tokenization.validate_case_matches_checkpoint(USE_LOWER, BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(
  vocab_file=BERT_VOCAB,
  do_lower_case=USE_LOWER
)

def parseXY(x, y):
    X, Y = [], []
    for i in tqdm(range(len(x))):
        left = x[i]
        right = y[i]
    
        # The beginning of a sentence, CLS stands thereby for CLASSIFICATION
        bert_tokens = ['[CLS]']
        # The padding
        y_ = ['PAD']
    
        for no, orig_token in enumerate(left):
            y_.append(right[no])
            t = tokenizer.tokenize(orig_token)
            bert_tokens.extend(t)
            y_.extend(['PAD']*(len(t)-1))
        # The SEP is required for the separation of sentences
        bert_tokens.append("[SEP]")
    
        y_.append("PAD")
        X.append(tokenizer.convert_tokens_to_ids(bert_tokens))
        Y.append([tag2idx[i] for i in y_])
  
    return X, Y


train_X, train_Y = parseXY(x_train, y_train)
test_X, test_Y = parseXY(x_test, y_test)



train_X = keras.preprocessing.sequence.pad_sequences(train_X, padding='post')
train_Y = keras.preprocessing.sequence.pad_sequences(train_Y, padding='post')

test_X = keras.preprocessing.sequence.pad_sequences(test_X, padding='post')
test_Y = keras.preprocessing.sequence.pad_sequences(test_Y, padding='post')

num_train_steps = int(len(train_X) / batch_size* num_epochs)
num_warmup_steps = int(num_train_steps*warmup_proportion)

bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)


#
# The model
#
class BERTEntityTagger(object):
  
    def __init__(
      self,
      dimension_output,
      learning_rate=2e-5
    ):
        self.X = tf.placeholder(tf.int32, shape=[None, None])
        self.Y = tf.placeholder(tf.int32, shape=[None, None])
    
        self.maxlen = tf.shape(self.X)[1]
        # We can evaluate the length this way because PAD is 0
        self.lengths = tf.count_nonzero(self.X, 1)
    
        #
        # The procedure of finetuning bert is:
        # - Create the default BERT model
        # - Fetch the output layer and pass it to a dense layer, with output dim #TAGS
        #
        model = modeling.BertModel(
          config=bert_config,
          is_training=True,
          input_ids=self.X,
          use_one_hot_embeddings=False
        )
        output_layer = model.get_sequence_output()
    
        logits = tf.layers.dense(output_layer, dimension_output)
    
        #
        # This is specific for training, as one can see we are using CRF
        #
        y_t = self.Y
    
        llh, transitions = tf.contrib.crf.crf_log_likelihood(logits, y_t, self.lengths)
    
        self.loss = tf.reduce_mean(-llh)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
    
        mask = tf.sequence_mask(self.lengths, maxlen=self.maxlen)
        self.tags_seq, _ = tf.contrib.crf.crf_decode(logits, transitions, self.lengths)
        self.tags_seq = tf.identity(self.tags_seq, name='logits')
    
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(self.tags_seq, mask)
    
        mask_label = tf.boolean_mask(y_t, mask)
    
        correct_prediction = tf.equal(self.prediction, mask_label)
    
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
#
# TRAINING
#
dimension_output = len(tag2idx)

tf.reset_default_graph()
sess = tf.InteractiveSession()

model = BERTEntityTagger(dimension_output, learning_rate=LR)

sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bert')

# load the pretrained model
saver = tf.train.Saver(var_list=var_lists)
saver.restore(sess, BERT_INIT_CHKPNT)

for e in range(num_epochs):
    start_time = time.time()
  
    #
    # Training
    #
    train_acc, train_loss, test_acc, test_loss = 0.0, 0.0, 0.0, 0.0
  
    progress = tqdm(range(0, len(train_X), batch_size), desc='training loop')
  
    for i in progress:
        batch_x = train_X[i:min(i+batch_size, train_X.shape[0])]
        batch_y = train_Y[i:min(i+batch_size, train_X.shape[0])]
    
        acc, loss, _ = sess.run(
          [model.accuracy, model.loss, model.optimizer],
          feed_dict = { model.X: batch_x, model.Y: batch_y}
        )
    
        assert not np.isnan(loss)
    
        train_loss += loss
        train_acc += acc
        progress.set_postfix(loss=loss, accuracy=acc)
  
    #
    # Testing
    #
    progress = tqdm(range(0, len(test_X), batch_size), desc='test loop')
  
    for i in progress:
        batch_x = test_X[i:min(i+batch_size, test_X.shape[0])]
        batch_y = test_Y[i:min(i+batch_size, test_X.shape[0])]
    
        acc, loss = sess.run(
          [model.accuracy, model.loss],
          feed_dict = { model.X: batch_x, model.Y: batch_y}
        )
    
        assert not np.isnan(loss)
    
        test_loss += loss
        test_acc += acc
        progress.set_postfix(loss=loss, accuracy=acc)
  
    train_loss /= len(train_X) / batch_size
    train_acc /= len(train_X) / batch_size
    test_loss /= len(test_X) / batch_size
    test_acc /= len(test_X) / batch_size
  
    print("Whole epoch took {} s".format(time.time() - start_time))
  
    print("epoch {} values: {} training loss; {} training acc; {} test loss; {} test acc".format(
      e, train_loss, train_acc, test_loss, test_acc
    ))