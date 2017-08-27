from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import random
import argparse
import numpy as np
import _pickle as cPickle
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_runner

from experiment import generate_experiment_fn

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--train-file',
      help='training file',
      required=True)
  parser.add_argument(
      '--eval-file',
      help='evaluate file',
      required=True)
  parser.add_argument(
      '--vocab-file',
      help='vocabulary file',
      required=True)
  parser.add_argument(
      '--job-dir',
      help='Location to write checkpoints, summaries, and export models',
      required=True)
  parser.add_argument(
      '--num-epochs',
      help='Maximum number of epochs on which to train',
      type=int,
      default=1000)
  parser.add_argument(
      '--batch-size',
      help='Batch size for steps',
      type=int,
      default=32)
  parser.add_argument(
      '--seed',
      help='Random seed',
      type=int,
      default=random.randint(0, 2**32))

  args = parser.parse_args()

  # NOTE: set random seed if specified
  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

  tf.logging.set_verbosity(tf.logging.INFO)

  vocab = cPickle.load(open(args.vocab_file, 'rb'))

  experiment_fn = generate_experiment_fn(
      train_file=args.train_file,
      eval_file=args.eval_file,
      vocab_size=len(vocab),
      batch_size=args.batch_size,
      num_epochs=args.num_epochs,
      seed=args.seed)

  learn_runner.run(experiment_fn, args.job_dir)

if __name__ == '__main__':
   main()
