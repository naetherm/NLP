import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils import shuffle
import re
import time
import collections

tf.compat.v1.disable_v2_behavior()

HIDDEN_LAYER = 256
NUM_LAYERS = 2
EMBEDDING_SIZE = 128
LEARNING_RATE = 1e-2
BATCH_SIZE = 16
NUM_EPOCHS = 20


def generate_dataset(words, n_words, atleast=1):
  count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
  counter = collections.Counter(words).most_common(n_words)
  counter = [i for i in counter if i[1] >= atleast]
  count.extend(counter)
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
  _line = line.split(' +++$+++ ')
  if len(_line) == 5:
    id2line[_line[0]] = _line[4]

convs = []
for line in conv_lines[:-1]:
  _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
  convs.append(_line.split(','))

questions = []
answers = []

for conv in convs:
  for i in range(len(conv) - 1):
    questions.append(id2line[conv[i]])
    answers.append(id2line[conv[i + 1]])


def cleanup_and_expand_text(text):
  text = text.lower()
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"he's", "he is", text)
  text = re.sub(r"she's", "she is", text)
  text = re.sub(r"it's", "it is", text)
  text = re.sub(r"that's", "that is", text)
  text = re.sub(r"what's", "that is", text)
  text = re.sub(r"where's", "where is", text)
  text = re.sub(r"how's", "how is", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"can't", "cannot", text)
  text = re.sub(r"n't", " not", text)
  text = re.sub(r"n'", "ng", text)
  text = re.sub(r"'bout", "about", text)
  text = re.sub(r"'til", "until", text)
  text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
  return ' '.join([i.strip() for i in filter(None, text.split())])


clean_questions = []
for question in questions:
  clean_questions.append(cleanup_and_expand_text(question))

clean_answers = []
for answer in answers:
  clean_answers.append(cleanup_and_expand_text(answer))

min_line_length = 2
max_line_length = 5
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
  if min_line_length <= len(question.split()) <= max_line_length:
    short_questions_temp.append(question)
    short_answers_temp.append(clean_answers[i])
  i += 1

short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
  if min_line_length <= len(answer.split()) <= max_line_length:
    short_answers.append(answer)
    short_questions.append(short_questions_temp[i])
  i += 1

question_test = short_questions[500:550]
answer_test = short_answers[500:550]
short_questions = short_questions[:500]
short_answers = short_answers[:500]

# Generate the question dataset -> the input
concat_from = ' '.join(short_questions + question_test).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = generate_dataset(concat_from, vocabulary_size_from)

# Generate the answer dataset -> the target
concat_to = ' '.join(short_answers + answer_test).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = generate_dataset(concat_to, vocabulary_size_to)

# Shortcuts to functional tags
GO = dictionary_from['GO']
PAD = dictionary_from['PAD']
EOS = dictionary_from['EOS']
UNK = dictionary_from['UNK']

for i in range(len(short_answers)):
  short_answers[i] += ' EOS'

print("questions: {}".format(questions[:2]))
print("answers: {}".format(answers[:2]))

print("short_questions: {}".format(short_questions[:2]))
print("short_answers: {}".format(short_answers[:2]))


# Helper method for transforming dataset to indices
def str_idx(corpus, dic):
  X = []
  for i in corpus:
    ints = []
    for k in i.split():
      ints.append(dic.get(k, UNK))
    X.append(ints)
  return X


class LSTMChatbot(object):
  def __init__(self, hidden_size, num_layers, embedded_size,
               from_dict_size, to_dict_size, learning_rate, batch_size):
    super(LSTMChatbot, self).__init__()
    
    def cells(h_size, reuse=False):
      return tf.compat.v1.nn.rnn_cell.LSTMCell(h_size, reuse=reuse)
    
    self.X = tf.compat.v1.placeholder(tf.int32, [None, None])
    self.Y = tf.compat.v1.placeholder(tf.int32, [None, None])
    self.X_seq_len = tf.compat.v1.placeholder(tf.int32, [None])
    self.Y_seq_len = tf.compat.v1.placeholder(tf.int32, [None])
    batch_size = tf.shape(input=self.X)[0]
    
    encoder_embeddings = tf.Variable(tf.random.uniform([from_dict_size, embedded_size], -1, 1))
    encoder_embedded = tf.nn.embedding_lookup(params=encoder_embeddings, ids=self.X)
    main = tf.strided_slice(self.X, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
    decoder_embedded = tf.nn.embedding_lookup(params=encoder_embeddings, ids=decoder_input)
    
    # Multi layer architecture ->
    for l in range(num_layers):
      (o_fw, o_bw), (s_fw, s_bw) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
        cell_fw=cells(hidden_size // 2),
        cell_bw=cells(hidden_size // 2),
        inputs=encoder_embedded,
        sequence_length=self.X_seq_len,
        scope='birnn_L{}'.format(l),
        dtype=tf.float32
      )
      encoder_embedded = tf.concat((o_fw, o_bw), axis=2)
    
    s_c_bi = tf.concat((s_fw.c, s_bw.c), axis=-1)
    s_h_bi = tf.concat((s_fw.h, s_bw.h), axis=-1)
    bi_lstm_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=s_c_bi, h=s_h_bi)
    s_last = tuple([bi_lstm_state] * num_layers)
    
    with tf.compat.v1.variable_scope("decoder"):
      rnn_cells_dec = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cells(hidden_size) for _ in range(num_layers)])
      outputs, _ = tf.compat.v1.nn.dynamic_rnn(rnn_cells_dec, decoder_embedded,
                                               sequence_length=self.X_seq_len,
                                               initial_state=s_last,
                                               dtype=tf.float32)
    self.logits = tf.compat.v1.layers.dense(outputs, to_dict_size)
    masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(input_tensor=self.Y_seq_len), dtype=tf.float32)
    self.cost = tfa.seq2seq.sequence_loss(logits=self.logits,
                                          targets=self.Y,
                                          weights=masks)
    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    y_t = tf.argmax(input=self.logits, axis=2)
    y_t = tf.cast(y_t, tf.int32)
    self.prediction = tf.boolean_mask(tensor=y_t, mask=masks)
    mask_label = tf.boolean_mask(tensor=self.Y, mask=masks)
    correct_pred = tf.equal(self.prediction, mask_label)
    self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))


# Start the session (this is < TF 2.0)
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()
model = LSTMChatbot(
  HIDDEN_LAYER,
  NUM_LAYERS,
  EMBEDDING_SIZE,
  len(dictionary_from),
  len(dictionary_to),
  LEARNING_RATE,
  BATCH_SIZE)
sess.run(tf.compat.v1.global_variables_initializer())

# Transform input and output to indices
X = str_idx(short_questions, dictionary_from)
Y = str_idx(short_answers, dictionary_to)
X_test = str_idx(question_test, dictionary_from)
Y_test = str_idx(answer_test, dictionary_from)

# For batching the input and target we will need the max. length of each
maxlen_question = max([len(x) for x in X]) * 2
maxlen_answer = max([len(y) for y in Y]) * 2
# For simplicity let's use the longest length
maxlen = max(maxlen_question, maxlen_answer)


# Batching input and target sequences according to the max. length of each
def pad_sentence_batch(sentence_batch, pad_int, maxlen):
  padded_seqs = []
  seq_lens = []
  max_sentence_len = maxlen
  for sentence in sentence_batch:
    padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
    seq_lens.append(maxlen)
  return padded_seqs, seq_lens


# Training
for i in range(NUM_EPOCHS):
  start_time = time.time()
  total_loss, total_accuracy = 0, 0
  X, Y = shuffle(X, Y)
  for k in range(0, len(short_questions), BATCH_SIZE):
    index = min(k + BATCH_SIZE, len(short_questions))
    batch_x, seq_x = pad_sentence_batch(X[k: index], PAD, maxlen)
    batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD, maxlen)
    predicted, accuracy, loss, _ = sess.run(
      [
        tf.argmax(input=model.logits, axis=2),
        model.accuracy,
        model.cost,
        model.optimizer
      ],
      feed_dict={
        model.X: batch_x,
        model.Y: batch_y,
        model.X_seq_len: seq_x,
        model.Y_seq_len: seq_y
      }
    )
    total_loss += loss
    total_accuracy += accuracy
    print('REAL ANSWER:', ' '.join([rev_dictionary_to[n] for n in batch_y[0] if n not in [0, 1, 2, 3]]))
    print('PREDICTED ANSWER:', ' '.join([rev_dictionary_to[n] for n in predicted[0] if n not in [0, 1, 2, 3]]), '\n')
  total_loss /= (len(short_questions) / BATCH_SIZE)
  total_accuracy /= (len(short_questions) / BATCH_SIZE)
  diff_time = time.time() - start_time
  print('[%f seconds] epoch: %d, avg loss: %f, avg accuracy: %f' % (diff_time, i + 1, total_loss, total_accuracy))

# Testing
batch_x, seq_x = pad_sentence_batch(X_test[:BATCH_SIZE], PAD, maxlen)
batch_y, seq_y = pad_sentence_batch(Y_test[:BATCH_SIZE], PAD, maxlen)
predicted = sess.run(tf.argmax(input=model.logits, axis=2), feed_dict={model.X: batch_x, model.X_seq_len: seq_x})

# Results
print("########################")
print("# TESTING")
print("########################")
for i in range(len(batch_x)):
  print('row %d' % (i + 1))
  print('QUESTION:', ' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]]))
  print('REAL ANSWER:', ' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in [0, 1, 2, 3]]))
  print('PREDICTED ANSWER:', ' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in [0, 1, 2, 3]]), '\n')
