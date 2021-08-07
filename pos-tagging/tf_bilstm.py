# First, download the required data
# !wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train
# !wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa

import re
import numpy as np
import tensorflow
from tensorflow.keras import backend as K
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


model = Sequential()

# Add Embedding layer
model.add(Embedding(input_dim=len(word2idx), output_dim=output_dim, input_length=len(train_X[0])))

# Add bidirectional LSTM
model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))

# Add LSTM
model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

# Add timeDistributed Layer
model.add(TimeDistributed(Dense(len(tag2idx), activation="relu")))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()




def train_model(X, y, model):
    loss = list()
    for i in range(25):
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss
    
train_model(train_X, train_Y, model)
