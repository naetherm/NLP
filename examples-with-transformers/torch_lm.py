import math
import copy
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


EMBEDDING_DIM = 200
HIDDEN_DIM = 200
NUM_LAYERS = 2
NUM_HEADS = 2
DROPOUT = 0.2
BATCH_SIZE = 20
BATCH_SIZE = 10
BPTT = 35
EPOCHS = 3
LEARNING_RATE = 5.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_INTERVAL = 200
WEIGHT_INIT_RANGE = 0.1


class TransformerModel(nn.Module):

    def __init__(self, num_token, d_model, num_head, d_hid, num_layers, dropout = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, num_head, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(num_token, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, num_token)

        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.uniform_(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


train_iter = WikiText103(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>']) 

def data_processing(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText103()
train_data = data_processing(train_iter)
val_data = data_processing(val_iter)
test_data = data_processing(test_iter)

def gen_all_batches(data):
    seq_len = data.size(0) // BATCH_SIZE
    data = data[:seq_len * BATCH_SIZE]
    data = data.view(BATCH_SIZE, seq_len).t().contiguous()
    return data.to(DEVICE)

train_data = gen_all_batches(train_data)
val_data = gen_all_batches(val_data)
test_data = gen_all_batches(test_data)

def get_batch(source, i):
    seq_len = min(BPTT, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

num_tokens = len(vocab)

model = TransformerModel(num_tokens, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
model_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train(model):
    model.train()
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(BPTT).to(DEVICE)

    num_batches = len(train_data) // BPTT
    for batch, i in enumerate(range(0, train_data.size(0) - 1, BPTT)):
        data, targets = get_batch(train_data, i)
        BATCH_SIZE = data.size(0)
        if BATCH_SIZE != BPTT:
            src_mask = src_mask[:BATCH_SIZE, :BATCH_SIZE]
        output = model(data, src_mask)
        loss = model_loss(output.view(-1, num_tokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss / LOG_INTERVAL
            ppl = math.exp(cur_loss)
            print(f'>> epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0

def evaluate(model, eval_data):
    model.eval()
    total_loss = 0.0
    src_mask = generate_square_subsequent_mask(BPTT).to(DEVICE)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, BPTT):
            data, targets = get_batch(eval_data, i)
            BATCH_SIZE = data.size(0)
            if BATCH_SIZE != BPTT:
                src_mask = src_mask[:BATCH_SIZE, :BATCH_SIZE]
            output = model(data, src_mask)
            output_flat = output.view(-1, num_tokens)
            total_loss += BATCH_SIZE * model_loss(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

best_val_loss = float('inf')
best_model = None

for epoch in range(1, EPOCHS + 1):
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    print(f'>> end of epoch {epoch:3d} | valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)


test_loss = evaluate(best_model, test_data)
test_ppl = math.exp(test_loss)
print(f'>> test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}')