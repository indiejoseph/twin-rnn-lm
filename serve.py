from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def generate_serving_input_fn():
  def _serving_input_fn():
    inputs = tf.placeholder(shape=[2], dtype=tf.float32)

    feature_placeholders = {
        'inputs': inputs
    }

    features = {
        key: tf.expand_dims(tensor, axis=0) for key, tensor in feature_placeholders.items()
    }

    input_fn_ops = tf.contrib.learn.utils.input_fn_utils.InputFnOps(features=features,
                                                                    labels=None,
                                                                    default_inputs=feature_placeholders)

    return input_fn_ops

  return _serving_input_fn
