
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from dataloader import retrieve_data
from attention_gru_cell import AttentionGRUCell

LOGGER = logging.getLogger(__name__)

epoch = 10
batch_size = 64
size_layer = 64
dropout_rate = 0.1
n_hops = 2

LOGGER.debug(">> Importing data ...")
train_data, test_data = retrieve_data()
LOGGER.debug(">> done.")

START = train_data.params['<start>']
END = train_data.params['<end>']


def shift_right(x):
  batch_size = tf.shape(x)[0]
  start = tf.compat.v1.to_int32(tf.fill([batch_size, 1], START))
  return tf.compat.v1.concat([start, x[:, :-1]], 1)


def GRU(name, rnn_size=None):
  return tf.compat.v1.nn.rnn_cell.GRUCell(
    rnn_size, kernel_initializer=tf.compat.v1.orthogonal_initializer(), name=name)


def position_encoding(sentence_size, embedding_size):
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return np.transpose(encoding)


def gen_episode(inputs_len, memory, q_vec, fact_vecs, proj_1, proj_2, attn_gru, is_training):
  def gen_attn(fact_vec):
    features = [fact_vec * q_vec,
                fact_vec * memory,
                tf.compat.v1.abs(fact_vec - q_vec),
                tf.compat.v1.abs(fact_vec - memory)]
    feature_vec = tf.compat.v1.concat(features, 1)
    attention = proj_1(feature_vec)
    attention = proj_2(attention)
    return tf.compat.v1.squeeze(attention, 1)
  
  attns = tf.compat.v1.map_fn(gen_attn, tf.compat.v1.transpose(fact_vecs, [1, 0, 2]))
  attns = tf.compat.v1.transpose(attns)
  attns = tf.compat.v1.nn.softmax(attns)
  attns = tf.compat.v1.expand_dims(attns, -1)
  
  _, episode = tf.compat.v1.nn.dynamic_rnn(
    attn_gru,
    tf.compat.v1.concat([fact_vecs, attns], 2),
    inputs_len,
    dtype=np.float32)
  return episode

class QA(object):
  def __init__(self, vocab_size):
    self.questions = tf.compat.v1.placeholder(tf.int32, [None, None])
    self.inputs = tf.compat.v1.placeholder(tf.int32, [None, None, None])
    self.questions_len = tf.compat.v1.placeholder(tf.int32, [None])
    self.inputs_len = tf.compat.v1.placeholder(tf.int32, [None])
    self.answers_len = tf.compat.v1.placeholder(tf.int32, [None])
    self.answers = tf.compat.v1.placeholder(tf.int32, [None, None])
    self.training = tf.compat.v1.placeholder(tf.bool)
    max_sent_len = train_data.params['max_sent_len']
    max_quest_len = train_data.params['max_quest_len']
    max_answer_len = train_data.params['max_answer_len']
  
    lookup_table = tf.compat.v1.get_variable('lookup_table', [vocab_size, size_layer], tf.float32)
    lookup_table = tf.compat.v1.concat((tf.compat.v1.zeros([1, size_layer]), lookup_table[1:, :]), axis=0)
  
    cell_fw = GRU('cell_fw', size_layer // 2)
    cell_bw = GRU('cell_bw', size_layer // 2)
    inputs = tf.compat.v1.nn.embedding_lookup(lookup_table, self.inputs)
    position = position_encoding(max_sent_len, size_layer)
    inputs = tf.compat.v1.reduce_sum(inputs * position, 2)
    birnn_out, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
      cell_fw,
      cell_bw,
      inputs,
      self.inputs_len,
      dtype=np.float32)
    fact_vecs = tf.compat.v1.concat(birnn_out, -1)
    fact_vecs = tf.compat.v1.layers.dropout(fact_vecs, dropout_rate, training=self.training)
  
    cell = GRU('question_rnn', size_layer)
    questions = tf.compat.v1.nn.embedding_lookup(lookup_table, self.questions)
    _, q_vec = tf.compat.v1.nn.dynamic_rnn(
      cell,
      questions,
      self.questions_len,
      dtype=np.float32)
  
    proj_1 = tf.compat.v1.layers.Dense(size_layer, tf.tanh, name='attn_proj_1')
    proj_2 = tf.compat.v1.layers.Dense(1, name='attn_proj_2')
    attn_gru = AttentionGRUCell(size_layer, name='attn_gru')
    memory_proj = tf.compat.v1.layers.Dense(size_layer, tf.nn.relu, name='memory_proj')
  
    memory = q_vec
    for i in range(n_hops):
      episode = gen_episode(self.inputs_len,
                            memory,
                            q_vec,
                            fact_vecs,
                            proj_1,
                            proj_2,
                            attn_gru,
                            self.training)
      memory = memory_proj(tf.compat.v1.concat([memory, episode, q_vec], 1))
  
    state_proj = tf.compat.v1.layers.Dense(size_layer, name='state_proj')
    vocab_proj = tf.compat.v1.layers.Dense(vocab_size, name='vocab_proj')
  
    memory = tf.compat.v1.layers.dropout(memory, dropout_rate, training=self.training)
    init_state = state_proj(tf.concat((memory, q_vec), -1))
    helper = tfa.seq2seq.TrainingSampler()# Sampler
      #inputs=tf.compat.v1.nn.embedding_lookup(lookup_table, shift_right(self.answers)),
      #sequence_length=self.answers_len)
    decoder = tfa.seq2seq.BasicDecoder(
      cell=cell,
      sampler=helper,
      #initial_state=init_state,
      output_layer=vocab_proj)
    decoder_output, _, _ = tfa.seq2seq.dynamic_decode(
      decoder=decoder)
    self.training_id = decoder_output.sample_id
    self.training_logits = decoder_output.rnn_output
  
    helper = tfa.seq2seq.GreedyEmbeddingSampler()
    #helper.initialize( # Sampler
    #  embedding=lookup_table,
    #  start_tokens=tf.tile(
    #    tf.constant([START], dtype=tf.int32), [tf.shape(init_state)[0]]),
    #  end_token=END)
    decoder = tfa.seq2seq.BasicDecoder(
      cell=cell,
      sampler=helper,
      #initial_state=init_state,
      output_layer=vocab_proj)
    decoder_output, _, _ = tfa.seq2seq.dynamic_decode(
      decoder=decoder,
      maximum_iterations=max_answer_len)
    self.predict_id = decoder_output.sample_id
  
    self.cost = tf.compat.v1.reduce_mean(tfa.seq2seq.sequence_loss(
      logits=self.training_logits,
      targets=self.answers,
      weights=tf.compat.v1.ones_like(self.answers, tf.float32)))
    self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.cost)
  
    correct_pred = tf.compat.v1.equal(self.training_id, self.answers)
    self.accuracy = tf.compat.v1.reduce_mean(tf.cast(correct_pred, tf.float32))


tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()
model = QA(train_data.params['vocab_size'])
sess.run(tf.compat.v1.global_variables_initializer())