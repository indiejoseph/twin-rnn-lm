from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.rnn import DeviceWrapper, MultiRNNCell
from ran_cell import RANCell
from utils import SwitchableDropoutWrapper

class TwinRNN(object):
  def __init__(self, configs, mode):
    self._configs = configs
    self.is_training = tf.constant(mode == tf.contrib.learn.ModeKeys.INFER)
    self.embedding = tf.get_variable('embedding', [configs['vocab_size'], configs['num_units']])

    # build graphs
    self.fw_cell = self.create_cell(0)
    self.bw_cell = self.create_cell(1)
    self.affine_w = tf.get_variable('affine_w', [self._configs['rnn_size'], self._configs['rnn_size']])
    self.affine_b = tf.get_variable('affine_b', [self._configs['rnn_size']])
    self.softmax_w = tf.get_variable('softmax_w', [configs['rnn_size'], configs['vocab_size']],
                                     initializer=tf.truncated_normal_initializer(stddev=1e-4))
    self.softmax_b = tf.get_variable('softmax_b', [configs['vocab_size']])

  def create_cell(self, device):
    cell = RANCell(self._configs['rnn_size'], normalize=self.is_training)
    cell = DeviceWrapper(cell, device='/gpu:{}'.format(device))
    cell = SwitchableDropoutWrapper(cell, output_keep_prob=self._configs['keep_prob'],
                                    variational_recurrent=True, is_train=self.is_training, dtype=tf.float32)

    return cell

  def forward_rnn(self, inputs, initial_state=None):
    input_arr = [tf.squeeze(inp, 1) for inp in tf.split(value=inputs, num_or_size_splits=self._configs['max_seq_length'], axis=1)]
    state = initial_state or self.fw_cell.zero_state(self._configs['batch_size'], tf.float32)
    outputs = []
    states = []

    for inp in input_arr:
      out, state = self.fw_cell(inp, state)
      outputs.append(out)
      states.append(state)
    outputs = tf.reshape(tf.concat(tf.stack(outputs), 1), [-1, self._configs['rnn_size']])
    return outputs, tf.stack(states)

  def backward_rnn(self, inputs, initial_state=None):
    input_rev = tf.reverse(inputs, axis=[1])
    input_arr = [tf.squeeze(inp, 1)
                 for inp in tf.split(value=input_rev, num_or_size_splits=self._configs['max_seq_length'], axis=1)]
    state = initial_state or self.bw_cell.zero_state(self._configs['batch_size'], tf.float32)
    outputs = []
    states = []
    for inp in input_arr:
      out, state = self.bw_cell(inp, state)
      outputs.append(out)
      states.append(state)
    outputs = tf.reshape(tf.concat(tf.stack(outputs), 1), [-1, self._configs['rnn_size']])
    return outputs, tf.stack(states)

  def get_logits(self, input_data, fw_initial_state=None, bw_initial_state=None):
    with tf.device('/cpu:0'):
      inputs = tf.nn.embedding_lookup(self.embedding, input_data)
      inputs = tf.contrib.layers.dropout(inputs, self._configs['keep_prob'], is_training=self.is_training)

    fw_outputs, fw_states = self.forward_rnn(inputs, fw_initial_state)
    bw_outputs, bw_states = self.backward_rnn(inputs, bw_initial_state)

    with tf.variable_scope('logits'):
      fw_logits = tf.matmul(fw_outputs, self.softmax_w) + self.softmax_b
      bw_logits = tf.matmul(tf.reverse(bw_outputs, axis=[1]), self.softmax_w) + self.softmax_b

    return fw_logits, bw_logits, fw_states, bw_states

  def get_predictions(self, logits):
    predictions = {
        'prediction': tf.nn.softmax(logits)
    }

    return predictions

  def get_loss(self, fw_logits, bw_logits, fw_states, bw_states, targets, mode=tf.contrib.learn.ModeKeys.TRAIN):
    # NOTE: `loss` should be `None` during the "infer" mode
    loss = None
    if mode == tf.contrib.learn.ModeKeys.INFER:
      return loss

    with tf.variable_scope('seq_loss'):
      flat_targets = tf.reshape(tf.concat(targets, 1), [-1])
      fw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fw_logits, labels=flat_targets)
      bw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=bw_logits, labels=flat_targets)

    with tf.variable_scope('penalty'):
      fw_flatted_states = tf.reshape(fw_states, [-1, self._configs['rnn_size']])
      bw_flatted_states = tf.reshape(bw_states, [-1, self._configs['rnn_size']])
      affine_transformation = (tf.matmul(fw_flatted_states, self.affine_w) + self.affine_b)
      l2_loss = tf.nn.l2_loss(affine_transformation - bw_flatted_states)

    loss = fw_loss + bw_loss + l2_loss
    perplexity = tf.exp(tf.reduce_mean(loss), name='perplexity')

    return tf.reduce_mean(loss), perplexity

def get_train_op(loss, params, mode):
  """
  Define the training operation which will be used to optimize the model.
  Uses [`tf.contrib.layers.optimize_loss`](https://goo.gl/z1PswO).
  """

  # NOTE: `train_op` should be `None` outside of the "train" mode
  train_op = None
  if mode != tf.contrib.learn.ModeKeys.TRAIN:
      return train_op

  global_step = tf.contrib.framework.get_or_create_global_step()

  learning_rate = tf.train.exponential_decay(
      learning_rate=params['learning_rate'],
      global_step=global_step,
      decay_steps=100000,
      decay_rate=0.96
  )

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=global_step,
      optimizer='SGD',
      learning_rate=learning_rate,
      clip_gradients=params['clip_grads'])

  return train_op

def model_fn(features, targets, mode, params):
  inputs = features['inputs']
  model = TwinRNN(params, (mode == tf.contrib.learn.ModeKeys.TRAIN))
  fw_logits, bw_logits, fw_states, bw_states = model.get_logits(inputs)
  predictions = model.get_predictions(fw_logits)
  loss, perplexity = model.get_loss(fw_logits, bw_logits, fw_states, bw_states, targets, mode)
  train_op = get_train_op(loss, params, mode)

  return tf.contrib.learn.ModelFnOps(predictions=predictions,
                                     loss=loss,
                                     train_op=train_op,
                                     mode=mode)

if __name__ == '__main__':
  params = {
      'batch_size': 32,
      'learning_rate': 0.1,
      'num_units': 300,
      'rnn_size': 512,
      'max_seq_length': 80,
      'vocab_size': 1000,
      'keep_prob': 0.5,
      'clip_grads': 5.0,
  }

  inputs = tf.placeholder(tf.int32, [params['batch_size'], params['max_seq_length']], name='inputs')
  targets = tf.placeholder(tf.int32, [params['batch_size'], params['max_seq_length']], name='targets')

  model = TwinRNN(params, mode=tf.contrib.learn.ModeKeys.TRAIN)
  fw_logits, bw_logits, fw_states, bw_states = model.get_logits(inputs)
  loss, perplexity = model.get_loss(fw_logits, bw_logits, fw_states, bw_states, targets)

  print(loss)
