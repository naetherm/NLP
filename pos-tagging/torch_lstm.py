import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm


EPOCHS=20

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
    return X, np.array(Y)
    
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
torch.manual_seed(1)
output_dim = 64


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMModel(output_dim, output_dim, len(word2idx), len(tag2idx))
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(EPOCHS):
    for s, t in zip(train_X, train_Y):
        model.zero_grad()
        tag_scores = model(torch.tensor(s, dtype=torch.int32))
        loss = loss_function(tag_scores, torch.from_numpy(t))
        loss.backward()
        optimizer.step()