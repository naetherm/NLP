
import os
import sys
import argparse
import time
import tqdm
from collections import Counter

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _single
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


def make_positions(tensor, padding_idx):
  mask = tensor.ne(padding_idx).int()
  return (torch.cumsum(mask, dim=1).type_as(mask)*mask).long() + padding_idx

def strip_pad(tensor, pad):
  return tensor[tensor.ne(pad)]


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


class FSeqConvTBC(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, padding=0):

    super(FSeqConvTBC, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _single(kernel_size)
    self.padding = _single(padding)

    self.weight = torch.nn.Parameter(
      torch.Tensor(self.kernel_size[0], in_channels, out_channels)
    )
    self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

  def forward(self, input):
    return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding[0])

  def __repr__(self):
    s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size, padding={padding}')
    if self.bias is None:
      s += ', bias=False'
    s += ')'

    return s.format(name=self.__class__.__name__, **self.__dict__)


class LearnedPositionalEmbedding(nn.Embedding):

  def __init__(
    self,
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int
  ):
    super(LearnedPositionalEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx)

    if self.padding_idx is not None:
      self.max_positions = self.num_embeddings - self.padding_idx - 1
    else:
      self.max_positions = self.num_embeddings

  def forward(self, input, incremental_state=None, positions=None):

    #assert(
    #  (positions is None) or (self.padding_idx is None),
    #
    #), "If positions is pre-computed then padding_idx should not be set!"

    if positions is None:
      if incremental_state is not None:
        positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
      else:
        positions = make_positions(input, self.padding_idx)

    return super().forward(positions)


class SinusoidalPositionalEmbedding(nn.Module):
  """This module produces sinusoidal positional embeddings of any length.
  Padding symbols are ignored.
  """

  def __init__(self, embedding_dim, padding_idx, init_size=2048):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.weights = SinusoidalPositionalEmbedding.get_embedding(
      init_size, embedding_dim, padding_idx
    )
    self.register_buffer("_float_tensor", torch.FloatTensor(1))
    self.max_positions = int(1e5)

  @staticmethod
  def get_embedding(
    num_embeddings: int, embedding_dim: int, padding_idx: int = None
  ):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
      1
    ) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
      num_embeddings, -1
    )
    if embedding_dim % 2 == 1:
      # zero pad
      emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
      emb[padding_idx, :] = 0
    return emb

  def forward(
    self,
    input,
    incremental_state = None,
    timestep = None,
    positions = None,
  ):
    """Input is expected to be of size [bsz x seqlen]."""
    #bspair = torch.onnx.operators.shape_as_tensor(input)
    bsz, seq_len, _ = list(input.size())
    max_pos = self.padding_idx + 1 + seq_len
    if self.weights is None or max_pos > self.weights.size(0):
      # recompute/expand embeddings if needed
      self.weights = SinusoidalPositionalEmbedding.get_embedding(
        max_pos, self.embedding_dim, self.padding_idx
      )
    self.weights = self.weights.to(self._float_tensor)

    if incremental_state is not None:
      # positions is the same for every token when decoding a single step
      pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len

      return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

    positions = make_positions(
      input, self.padding_idx, onnx_trace=self.onnx_trace
    )

    return (
      self.weights.index_select(0, positions.view(-1))
      .view(bsz, seq_len, -1)
      .detach()
    )


def ConvEmbedding(num_embeddings, embedding_dim, padding_idx):
  m = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
  nn.init.normal_(m.weight, 0, 0.1)
  nn.init.constant_(m.weight[padding_idx], 0)
  return m


def PositionEmbedding(num_embeddings, embedding_dim, padding_idx, learn_embedding=True):
  if learn_embedding:
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
  else:
    m = SinusoidalPositionalEmbedding(embedding_dim=embedding_dim, padding_idx=padding_idx)
    return m


def Linear(in_features, out_features, dropout=0):
  """Weight-normalized Linear layer (input: N x T x C)"""
  m = nn.Linear(in_features, out_features)
  nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
  nn.init.constant_(m.bias, 0)
  return nn.utils.weight_norm(m)


def extend_conv_spec(convolutions):
  """
  Extends convolutional spec that is a list of tuples of 2 or 3 parameters
  (kernel size, dim size and optionally how many layers behind to look for residual)
  to default the residual propagation param if it is not specified
  """
  extended = []
  for spec in convolutions:
    if len(spec) == 3:
      extended.append(spec)
    elif len(spec) == 2:
      extended.append(spec + (1,))
    else:
      raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
  return tuple(extended)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
  m = FSeqConvTBC(in_channels, out_channels, kernel_size, **kwargs)
  std = math.sqrt((4*(1.0-dropout))/ (m.kernel_size[0] * in_channels))
  nn.init.normal_(m.weight, mean=0, std=std)
  nn.init.constant_(m.bias, 0)

  return nn.utils.weight_norm(m, dim=2)


class CNNModel(nn.Module):

  def __init__(
    self,
    embedding_dim,
    vocab_size,
    tags_size,
    padding_idx=0,
    dropout=0.1,
    max_positions=2048,
    bias=True,
    convolutions=((128, 3),)*2):
    super(CNNModel, self).__init__()

    # Specify
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.vocab_size = vocab_size
    self.tags_size = tags_size
    self.bias = bias
    self.dropout = dropout
    self.learn_embedding = True

    # Create the model structure
    self.embeddings = ConvEmbedding(self.vocab_size, self.embedding_dim, self.padding_idx)

    self.position_embeddings = PositionEmbedding(max_positions, self.embedding_dim, self.padding_idx, learn_embedding=self.learn_embedding)

    # Now the convolutional part
    convolutions = extend_conv_spec(convolutions)
    in_channels = convolutions[0][0]
    self.fc1 = Linear(self.embedding_dim, in_channels, dropout=dropout)
    self.projections = nn.ModuleList()
    self.convolutions = nn.ModuleList()

    self.residuals = []

    layer_in_channels = [in_channels]

    for _, (out_channels, kernel_size, residual) in enumerate(convolutions):

      if residual == 0:
        residual_dim = out_channels
      else:
        residual_dim = layer_in_channels[-residual]

      self.projections.append(
        Linear(residual_dim, out_channels) if residual_dim != out_channels else None
      )

      if kernel_size % 2 == 1:
        padding = kernel_size // 2
      else:
        padding = 0

      self.convolutions.append(
        ConvTBC(in_channels, out_channels*2, kernel_size, dropout=dropout, padding=padding)
      )
      self.residuals.append(residual)
      in_channels = out_channels
      layer_in_channels.append(out_channels)
    self.fc2 = Linear(in_channels, embedding_dim)

    # Last part: Linear projection back to the tagset
    self.hidden2tag = nn.Linear(embedding_dim, tags_size)

  def forward(self, X):
    """
    X -> the input sequence
    """
    # embed tokens and positions
    a = self.embeddings(X)
    b = self.position_embeddings(X)
    x = a + b
    x = F.dropout(x, p=self.dropout, training=True)

    input_embedding = x

    # Project to the convolution size
    x = self.fc1(x)

    # used to mask padding in the input
    encoder_padding_mask = X.eq(self.padding_idx).t() # BxT -> TxB
    if not encoder_padding_mask.any():
      encoder_padding_mask = None

    # BxTxC -> TxBxC
    x = x.transpose(0, 1)

    residuals = [x]

    # loop through all convolutions
    for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
      if res_layer > 0:
        residual = residuals[-res_layer]
        residual = residual if proj is None else proj(residual)
      else:
        residual = None

      if encoder_padding_mask is not None:
        x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

      x = F.dropout(x, p=self.dropout, training=True)

      if conv.kernel_size[0] % 2 == 1:
        x = conv(x)
      else:
        padding_l = (conv.kernel_size[0] - 1) // 2
        padding_r = conv.kernel_size[0] // 2

        x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
        x = conv(x)

      x = F.glu(x, dim=2)

      if residual is not None:
        x = (x + residual) * math.sqrt(0.5)

    # Back-transformation: TxBxC -> BxTxC
    x = x.transpose(1, 0)
    x = self.fc2(x)

    if encoder_padding_mask is not None:
      encoder_padding_mask = encoder_padding_mask.t() # TxB -> BxT
      x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

    y = (x + input_embedding) * math.sqrt(0.5)
    y = self.hidden2tag(y)
    y = F.log_softmax(y, dim=-1)

    # Done
    return y


idx2char, char2idx, max_line_length, total_length = lazy_scan_vocabulary(TRAINING_DATA)
vocab_size = len(idx2char) + 1
print(f"vocab size: {vocab_size}")

model = CNNModel(EMBEDDING_DIM, vocab_size, NUM_TAGS)


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
      get_batches(TRAINING_DATA, char2idx, True), desc="Training", total=total_length
    )
    for x, y in pbar:
      batch_x = x
      batch_y = y

      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)

      model.zero_grad()

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