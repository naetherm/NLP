
import os
import sys
import argparse
import time
import tqdm
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics


EMBEDDING_DIM = 256
HIDDEN_DIM = 128
NUM_TAGS = 2
TRAINING_DATA = "./oscar.atom.txt"
LR = 1e-3
NUM_EPOCHS = 10


def get_length_of_file_in_lines(text_file):
  file_len = 0
  with open(text_file, encoding='utf-8') as fin:
    for l in fin:
      file_len += 1
  return file_len


def lazy_scan_vocabulary(text_file, padding_idx=0, min_count=1):
  """
  Responsible for loading the whole vocabulary of the given training set.
  """
  max_line_length = -1
  total_length = 0
  counter = Counter()

  with open(text_file, encoding='utf-8') as fin:
    for line in tqdm.tqdm(fin, desc="Analyzing vocabulary"):
      text = line.rstrip()

      if len(line) > max_line_length:
        max_line_length = len(line)

      c2 = Counter(vocab for vocab in text)

      total_length += 1

      counter.update(c2)

  idx2char = []
  char2idx = {}
  idx2char.append("@@@PADDING@@@")
  char2idx["@@@PADDING@@@"] = padding_idx
  idx2char.append("@@@UNKNOWN@@@")
  char2idx["@@@UNKNOWN@@@"] = padding_idx+1
  idx2char.extend([vocab for vocab in sorted(counter, key=lambda x:-counter[x])])
  char2idx.update({vocab:idx for idx, vocab in enumerate(idx2char)})

  return idx2char, char2idx, max_line_length, total_length


def space_tag(sent, nonspace=0, space=1):
  """
  :param sent: str
      Input sentence
  :param nonspace: Object
      Non-space tag. Default is 0, int type
  :param space: Object
      Space tag. Default is 1, int type
  It returns
  ----------
  chars : list of character
  tags : list of tag
  (example)
      sent  = 'Test sentence for demonstration purposes.'
      chars = list('Testsentencefordemonstrationpurposes.')
      tags  = [{0,1},...]
  """

  sent = sent.strip()
  chars = list(sent.replace(' ',''))
  tags = [nonspace]*(len(chars) - 1) + [space]
  idx = 0

  for c in sent:
    if c == ' ':
      tags[idx-1] = space
    else:
      idx += 1

  return chars, tags


def to_idx(item, mapper, unknown="@@@UNKNOWN@@@"):
  """
  :param item: Object
      Object to be encoded
  :param mapper: dict
      Dictionary from item to idx
  :param unknown: int
      Index of unknown item. If None, use len(mapper)
  It returns
  ----------
  idx : int
      Index of item
  """

  if item in mapper.keys():
    return mapper[item]
  return mapper["@@@UNKNOWN@@@"]

  if unknown is None:
    unknown = len(mapper)
  return mapper.get(item, unknown)


def to_item(idx, idx_to_char, unknown='Unk'):
  """
  :param idx: int
      Index of item
  :param idx_to_char: list of Object
      Mapper from index to item object
  :param unknown: Object
      Return value when the idx is outbound of idx_to_char
      Default is 'Unk', str type
  It returns
  ----------
  object : object
      Item that corresponding idx
  """

  if 0 <= idx < len(idx_to_char):
    return idx_to_char[idx]
  return unknown


def sent_to_xy(sent, char_to_idx, cnn_mode=False):
  """
  :param sent: str
      Input sentence
  :param char_to_idx: dict
      Dictionary from character to index
  It returns
  ----------
  idxs : torch.LongTensor
      Encoded character sequence
  tags : torch.LongTensor
      Space tag sequence
  """

  chars, tags = space_tag(sent)
  if cnn_mode:
    idxs = torch.LongTensor(
      [[to_idx(c, char_to_idx) for c in chars]])
  else:
    idxs = torch.LongTensor(
      [to_idx(c, char_to_idx) for c in chars])
  tags = torch.LongTensor([tags])
  return idxs, tags


class LSTMModel(nn.Module):

  def __init__(
    self,
    embedding_dim,
    hidden_dim,
    vocab_size,
    tags_size,
    num_layers=1,
    bias=True,
    dropout=0.1,
    bidirectional=False):
    super(LSTMModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim,
      num_layers=num_layers,
      bias=bias,
      dropout=dropout,
      bidirectional=bidirectional)
    self.hidden2tag = nn.Linear(hidden_dim * (1 + self.bidirectional), tags_size)
    self.hidden = self.init_hidden() # hidden, cell

  def forward(self, char_idxs):
    self.hidden = self.init_hidden() # hidden, cell
    embeds = self.embeddings(char_idxs)
    lstm_out, self.hidden = self.lstm(
      embeds.view(len(char_idxs), 1, -1), self.hidden)
    tag_space = self.hidden2tag(lstm_out.view(char_idxs.size()[0], -1))
    tag_scores = F.log_softmax(tag_space, dim=1)

    return tag_scores

  def init_hidden(self):
    # (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(1 + self.bidirectional, 1, self.hidden_dim),
            torch.zeros(1 + self.bidirectional, 1, self.hidden_dim))


class GRUModel(nn.Module):

  def __init__(
    self,
    embedding_dim,
    hidden_dim,
    vocab_size,
    tags_size,
    num_layers=1,
    bias=True,
    dropout=0.1,
    bidirectional=False):
    super(GRUModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.GRU(embedding_dim, hidden_dim,
      num_layers=num_layers,
      bias=bias,
      dropout=dropout,
      bidirectional=bidirectional)
    self.hidden2tag = nn.Linear(hidden_dim * (1 + self.bidirectional), tags_size)
    self.hidden = self.init_hidden() # hidden, cell

  def forward(self, char_idxs):
    self.hidden = self.init_hidden() # hidden, cell
    embeds = self.embeddings(char_idxs)
    lstm_out, self.hidden = self.lstm(
      embeds.view(len(char_idxs), 1, -1), self.hidden)
    tag_space = self.hidden2tag(lstm_out.view(char_idxs.size()[0], -1))
    tag_scores = F.log_softmax(tag_space, dim=1)

    return tag_scores

  def init_hidden(self):
    # (num_layers, minibatch_size, hidden_dim)
    return torch.zeros(1 + self.bidirectional, 1, self.hidden_dim)


idx2char, char2idx, max_line_length, total_length = lazy_scan_vocabulary(TRAINING_DATA)
vocab_size = len(idx2char) + 1
print(f"vocab size: {vocab_size}")

model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, NUM_TAGS)


def train(model, char2idx, idx2char, total_length, use_gpu=False):
  print("Un train")
  batch_size = 1

  is_cuda = use_gpu and torch.cuda.is_available()
  device = torch.device("cuda" if is_cuda else "cpu")

  loss_function = nn.CrossEntropyLoss().to(device)
  optimizer = optim.Adam(model.parameters(), lr=LR)

  # Create training and testing data


  def get_batches(filename, indexer, cnn_mode, max_seq_length=2040, min_seq_length=4, batch_size=1):
    with open(filename, 'r', encoding='utf-8') as fin:
      for line in fin:
        line = line.strip()

        if min_seq_length < len(line) < max_seq_length:
          x, y = sent_to_xy(line, indexer, cnn_mode)

          yield (x, y)


  for e in range(NUM_EPOCHS-1):
    start_time = time.time()
    print("Epoch %d/%d" % (e, NUM_EPOCHS-1))

    # Training
    train_acc, train_loss = 0.0, 0.0
    trained_sample_length = 0

    pbar = tqdm.tqdm(
      get_batches(TRAINING_DATA, char2idx, False), desc="Training", total=total_length
    )
    for x, y in pbar:
      batch_x = x
      batch_y = y

      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)

      model.zero_grad()

      model.hidden = model.init_hidden()
      tag_scores = model(batch_x)
      tag_scores = tag_scores.squeeze()
      tags = batch_y.squeeze()

      try:
        loss = loss_function(tag_scores, tags)

        loss.backward()
        optimizer.step()

        if is_cuda:
          loss_value = loss.cpu().data.numpy()
          ts_value = tag_scores.cpu().data.numpy()
        else:
          loss_value = loss.data.numpy()
          ts_value = tag_scores.data.numpy()
        acc = (np.argmax(ts_value, -1) == np.asarray(tags)).sum().item() / len(tags)

        # Sanity check
        #assert not np.isnan(loss.numpy())
        train_loss += loss_value
        train_acc += acc
        pbar.set_postfix(loss=loss, accuracy=acc)

        trained_sample_length += 1

      except:
        print("tag_scores: {}".format(tag_scores))
        print("tags: {}".format(tags))


    print("Training: Loss={}, Acc={}".format(train_loss/trained_sample_length, train_acc/trained_sample_length))


train(model, char2idx, idx2char, total_length, False)