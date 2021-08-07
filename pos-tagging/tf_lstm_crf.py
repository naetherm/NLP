# First, download the required data
# !wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train
# !wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa

import re
import numpy as np
import tensorflow
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow_addons.text import crf_log_likelihood, crf_decode
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

def parse(file):
    with open(file) as fopen:
        texts = fopen.read().split('\n')
    left, right = [], []
    for text in texts:
        if '-DOCSTART-' in text or not len(text):
            continue
        splitted = text.split()
        left.append(splitted[0])
        right.append(splitted[1])
    return left, right
    
left_train, right_train = parse('eng.train')
left_test, right_test = parse('eng.testa')

def process_string(string):
    string = re.sub('[^A-Za-z0-9\-\/ ]+', ' ', string).split()
    return ' '.join([to_title(y.strip()) for y in string])

def to_title(string):
    if string.isupper():
        string = string.title()
    return string
    
word2idx = {'PAD': 0,'NUM':1,'UNK':2}
tag2idx = {'PAD': 0}
char2idx = {'PAD': 0}
word_idx = 3
tag_idx = 1
char_idx = 1

def parse_XY(texts, labels):
    global word2idx, tag2idx, char2idx, word_idx, tag_idx, char_idx
    X, Y = [], []
    for no, text in enumerate(texts):
        text = text.lower()
        tag = labels[no]
        for c in text:
            if c not in char2idx:
                char2idx[c] = char_idx
                char_idx += 1
        if tag not in tag2idx:
            tag2idx[tag] = tag_idx
            tag_idx += 1
        Y.append(tag2idx[tag])
        if text not in word2idx:
            word2idx[text] = word_idx
            word_idx += 1
        X.append(word2idx[text])
    return X, tensorflow.keras.utils.to_categorical(np.array(Y))
    
train_X, train_Y = parse_XY(left_train, right_train)
test_X, test_Y = parse_XY(left_test, right_test)

idx2word = {idx: tag for tag, idx in word2idx.items()}
idx2tag = {i: w for w, i in tag2idx.items()}

seq_len = 50
def iter_seq(x):
    return np.array([x[i: i+seq_len] for i in range(0, len(x)-seq_len, 1)])

def to_train_seq(*args):
    return [iter_seq(x) for x in args]

def generate_char_seq(batch):
    x = [[len(idx2word[i]) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((batch.shape[0],batch.shape[1],maxlen),dtype=np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(idx2word[batch[i,k]]):
                temp[i,k,-1-no] = char2idx[c]
    return temp
    

X_seq, Y_seq = to_train_seq(train_X, train_Y)
X_char_seq = generate_char_seq(X_seq)
X_seq.shape

X_seq_test, Y_seq_test = to_train_seq(test_X, test_Y)
X_char_seq_test = generate_char_seq(X_seq_test)
X_seq_test.shape

train_X, train_Y, train_char = X_seq, Y_seq, X_char_seq
test_X, test_Y, test_char = X_seq_test, Y_seq_test, X_char_seq_test


from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)
output_dim = 64

class CRF(L.Layer):
    def __init__(self,
                 output_dim,
                 sparse_target=True,
                 **kwargs):
        """    
        Args:
            output_dim (int): the number of labels to tag each temporal input.
            sparse_target (bool): whether the the ground-truth label represented in one-hot.
        Input shape:
            (batch_size, sentence length, output_dim)
        Output shape:
            (batch_size, sentence length, output_dim)
        """
        super(CRF, self).__init__(**kwargs)
        self.output_dim = int(output_dim) 
        self.sparse_target = sparse_target
        self.input_spec = L.InputSpec(min_ndim=3)
        self.supports_masking = False
        self.sequence_lengths = None
        self.transitions = None

    def build(self, input_shape):
        assert len(input_shape) == 3
        f_shape = tensorflow.TensorShape(input_shape)
        input_spec = L.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def call(self, inputs, sequence_lengths=None, training=None, **kwargs):
        sequences = tensorflow.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tensorflow.convert_to_tensor(sequence_lengths).dtype == 'int32'
            seq_len_shape = tensorflow.convert_to_tensor(sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            self.sequence_lengths = K.flatten(sequence_lengths)
        else:
            self.sequence_lengths = tensorflow.ones(tensorflow.shape(inputs)[0], dtype=tensorflow.int32) * (
                tensorflow.shape(inputs)[1]
            )

        viterbi_sequence, _ = crf_decode(sequences,
                                         self.transitions,
                                         self.sequence_lengths)
        output = K.one_hot(viterbi_sequence, self.output_dim)
        return K.in_train_phase(sequences, output)

    @property
    def loss(self):
        def crf_loss(y_true, y_pred):
            y_pred = tensorflow.convert_to_tensor(y_pred, dtype=self.dtype)
            log_likelihood, self.transitions = crf_log_likelihood(
                y_pred,
                tensorflow.cast(K.argmax(y_true), dtype=tensorflow.int32) if self.sparse_target else y_true,
                self.sequence_lengths,
                transition_params=self.transitions,
            )
            return tensorflow.reduce_mean(-log_likelihood)
        return crf_loss

    @property
    def accuracy(self):
        def viterbi_accuracy(y_true, y_pred):
            # -1e10 to avoid zero at sum(mask)
            mask = K.cast(
                K.all(K.greater(y_pred, -1e10), axis=2), K.floatx())
            shape = tensorflow.shape(y_pred)
            sequence_lengths = tensorflow.ones(shape[0], dtype=tensorflow.int32) * (shape[1])
            y_pred, _ = crf_decode(y_pred, self.transitions, sequence_lengths)
            if self.sparse_target:
                y_true = K.argmax(y_true, 2)
            y_pred = K.cast(y_pred, 'int32')
            y_true = K.cast(y_true, 'int32')
            corrects = K.cast(K.equal(y_true, y_pred), K.floatx())
            return K.sum(corrects * mask) / K.sum(mask)
        return viterbi_accuracy

    def compute_output_shape(self, input_shape):
        tensorflow.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'sparse_target': self.sparse_target,
            'supports_masking': self.supports_masking,
            'transitions': K.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(base_config, **config)

model = Sequential()

# Add Embedding layer
model.add(Embedding(input_dim=len(word2idx), output_dim=output_dim, input_length=len(train_X[0])))

# Add LSTM
model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

# Add timeDistributed Layer
model.add(TimeDistributed(Dense(len(tag2idx), activation="relu")))
crf = CRF(len(tag_index), sparse_target=True)
model.add(crf)

# Compile model
model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy])
model.summary()




def train_model(X, y, model):
    loss = list()
    for i in range(25):
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss
    
train_model(train_X, train_Y, model)
