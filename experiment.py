from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from inputs import generate_input_fn
from serve import generate_serving_input_fn
from model import model_fn

def generate_experiment_fn(train_file, eval_file, vocab_size, batch_size, num_epochs, seed):
  params = {
      'batch_size': batch_size,
      'learning_rate': 0.01,
      'num_units': 300,
      'rnn_size': 512,
      'max_seq_length': 80,
      'vocab_size': vocab_size,
      'keep_prob': 0.5,
      'clip_grads': 5.0,
  }

  def _experiment_fn(output_dir):
    train_input_fn = generate_input_fn(
        data_files=train_file,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=True)

    eval_input_fn = generate_input_fn(
        data_files=eval_file,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    run_config = tf.contrib.learn.RunConfig(tf_random_seed=seed)

    estimator = tf.contrib.learn.Estimator(
        model_dir=output_dir,
        model_fn=model_fn,
        config=run_config,
        params=params)

    eval_metrics = {
        'accuracy': tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                                prediction_key='prediction',
                                                label_key='target')
    }

    serving_input_fn = generate_serving_input_fn()

    export_strategy = tf.contrib.learn.utils.make_export_strategy(
        serving_input_fn=serving_input_fn,
        exports_to_keep=1)
    export_strategies = [export_strategy]

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        eval_metrics=eval_metrics,
        eval_steps=None,
        export_strategies=export_strategies
    )

    return experiment

  return _experiment_fn
