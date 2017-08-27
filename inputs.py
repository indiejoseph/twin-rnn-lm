from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def generate_input_fn(data_files, batch_size, num_epochs=None, threads=1, capacity=1000, shuffle=False):
  def _input_fn():
    # NOTE: always place the input pipeline on the CPU
    with tf.device('/cpu:0'):
      filenames = tf.train.match_filenames_once(data_files, name='train_filenames')

      filename_queue = tf.train.string_input_producer(
          filenames,
          num_epochs=num_epochs,
          shuffle=shuffle)

      _, serialized = tf.TFRecordReader().read(filename_queue)
      _, features = tf.parse_single_sequence_example(serialized=serialized, sequence_features={
          'inputs': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
          'targets': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
      })
      all_inputs = features['inputs']
      all_targets = features['targets']

      inputs_batch, targets_batch = tf.train.batch(
          tensors=[all_inputs, all_targets],
          batch_size=batch_size,
          num_threads=threads,
          capacity=capacity,
          dynamic_pad=True,
      )

      # NOTE: must be a dictionary for exporting
      features = {
          'inputs': inputs_batch
      }

      return features, targets_batch

  return _input_fn
