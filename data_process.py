# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import math
import codecs
from tqdm import tqdm
import collections
import _pickle as cPickle
import tensorflow as tf
import numpy as np

PAD = '_PAD'
GO = '_GO'
EOS = '_EOS'
UNK = '_UNK'
SPACE = ' '
NEW_LINE = '\n'
START_VOCAB = [PAD, GO, EOS, UNK, SPACE, NEW_LINE]

def normalize_unicodes(text):
  text = normalize_punctuation(text)
  text = "".join([Q2B(c) for c in list(text)])
  return text


def replace_all(repls, text):
  # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], text)
  return re.sub(u'|'.join(re.escape(key) for key in repls.keys()),
                lambda k: repls[k.group(0)], text)


def normalize_punctuation(text):
  cpun = [['	'],
          [u'﹗'],
          [u'“', u'゛', u'〃', u'′'],
          [u'”'],
          [u'´', u'‘', u'’'],
          [u'；', u'﹔'],
          [u'《', u'〈', u'＜'],
          [u'》', u'〉', u'＞'],
          [u'﹑'],
          [u'【', u'『', u'〔', u'﹝', u'｢', u'﹁'],
          [u'】', u'』', u'〕', u'﹞', u'｣', u'﹂'],
          [u'（', u'「'],
          [u'）', u'」'],
          [u'﹖'],
          [u'︰', u'﹕'],
          [u'・', u'．', u'·', u'‧', u'°'],
          [u'●', u'○', u'▲', u'◎', u'◇', u'■', u'□', u'※', u'◆'],
          [u'〜', u'～', u'∼'],
          [u'︱', u'│', u'┼'],
          [u'╱'],
          [u'╲'],
          [u'—', u'ー', u'―', u'‐', u'−', u'─', u'﹣', u'–', u'ㄧ']]
  epun = [u' ', u'！', u'"', u'"', u'\'', u';', u'<', u'>', u'、',
          u'[', u']', u'(', u')', u'？', u'：', u'･', u'•', u'~',
          u'|', u'/', u'\\', u'-']
  repls = {}

  for i, pun_list in enumerate(cpun):
    for _, pun in enumerate(pun_list):
      repls[pun] = epun[i]

  return replace_all(repls, text)


def Q2B(uchar):
  """全角转半角"""
  inside_code = ord(uchar)
  if inside_code == 0x3000:
    inside_code = 0x0020
  else:
    inside_code -= 0xfee0
  # 转完之后不是半角字符返回原来的字符
  if inside_code < 0x0020 or inside_code > 0x7e:
    return uchar
  return chr(inside_code)

def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def convert_to_records(inputs, targets, filename):
  writer = tf.python_io.TFRecordWriter(filename)

  for i in tqdm(range(inputs.shape[0])):
    feature_list = {
        'inputs': _int64_feature_list(inputs[i]),
        'targets': _int64_feature_list(targets[i])
    }

    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    sequence_example = tf.train.SequenceExample(feature_lists=feature_lists)
    writer.write(sequence_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  data_dir = './data/news'
  max_seq_length = 80
  vocab_file = os.path.join(data_dir, 'vocab.pkl')
  input_file = os.path.join(data_dir, 'input.txt')
  train_file = os.path.join(data_dir, 'train.tfrecord')
  valid_file = os.path.join(data_dir, 'valid.tfrecord')

  with codecs.open(input_file, 'r', encoding='utf-8') as f:
    raw_text = f.read()
    raw_text = normalize_unicodes(raw_text)

  counter = collections.Counter(raw_text)
  count_pairs = sorted(counter.items(), key=lambda x: -x[1])

  threshold = 10 # remove rare word
  chars, counts = zip(*count_pairs)
  chars = START_VOCAB + [c for i, c in enumerate(chars) if c not in START_VOCAB and counts[i] > threshold]
  vocab_size = len(chars)
  vocab = dict(zip(chars, range(len(chars))))

  # export vocab file
  with open(vocab_file, 'wb') as f:
    cPickle.dump(chars, f)

  unk_index = START_VOCAB.index(UNK)
  tensor = np.array([vocab.get(c, unk_index) for c in raw_text], dtype=np.int64)

  # 90% as train data
  data_size = tensor.shape[0]
  train_size = math.floor(int(data_size * 0.9) / max_seq_length) * max_seq_length
  valid_size = math.floor((data_size - train_size) / max_seq_length) * max_seq_length
  valid = tensor[train_size:train_size + valid_size]
  train = tensor[:train_size]

  xdata = train
  ydata = np.copy(train)
  ydata[:-1] = xdata[1:]
  ydata[-1] = xdata[0]
  xdata = xdata.reshape([int(xdata.shape[0] / max_seq_length), -1])
  ydata = ydata.reshape([int(ydata.shape[0] / max_seq_length), -1])

  x_valid = valid
  y_valid = np.copy(valid)
  y_valid[:-1] = x_valid[1:]
  y_valid[-1] = x_valid[0]
  x_valid = x_valid.reshape([int(x_valid.shape[0] / max_seq_length), -1])
  y_valid = y_valid.reshape([int(y_valid.shape[0] / max_seq_length), -1])

  convert_to_records(xdata, ydata, train_file)
  convert_to_records(x_valid, y_valid, valid_file)
