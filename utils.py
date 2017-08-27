from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn import DropoutWrapper

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def linear(args,
           output_size,
           bias,
           bias_initializer=None,
           kernel_initializer=None,
           kernel_regularizer=None,
           bias_regularizer=None,
           normalize=False):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
    kernel_regularizer: kernel regularizer
    bias_regularizer: bias regularizer
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer,
        regularizer=kernel_regularizer)

    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)

    res = tf.cond(normalize, lambda: tf.contrib.layers.layer_norm(res), lambda: res)

    # remove the layer's bias if there is one (because it would be redundant)
    if not bias:
      return res

    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer,
          regularizer=bias_regularizer)

  return nn_ops.bias_add(res, biases)


class SwitchableDropoutWrapper(DropoutWrapper):
  def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0,
               variational_recurrent=False, seed=None, dtype=None):
    super(SwitchableDropoutWrapper, self).__init__(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob,
                                                   variational_recurrent=variational_recurrent, seed=seed, dtype=dtype)
    self.is_train = is_train

  def __call__(self, inputs, state, scope=None):
    outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
    tf.get_variable_scope().reuse_variables()
    outputs, new_state = self._cell(inputs, state, scope)
    outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
    if isinstance(state, tuple):
      new_state = state.__class__(*[tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                    for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
    else:
      new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
    return outputs, new_state
