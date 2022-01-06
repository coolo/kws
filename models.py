# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications Copyright 2017 Arm Inc. All Rights Reserved.
# Added new model definitions for speech command recognition used in
# the paper: https://arxiv.org/pdf/1711.07128.pdf
#
#

"""Model definitions for simple speech recognition.

"""
import math

import tensorflow as tf
from tf_slim import layers as slayers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

WINDOW_SIZE_MS = 20
WINDOW_STRIDE_MS = 10
SAMPLE_RATE = 16000

def prepare_model_settings(dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
     dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = SAMPLE_RATE
  window_size_samples = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
  window_stride_samples = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  return {
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count
  }

def create_model(fingerprint_4d, model_settings, is_training):
  """Builds a model with convolutional recurrent networks with GRUs
  Based on the model definition in https://arxiv.org/abs/1703.05390
  model_size_info: defines the following convolution layer parameters
      {number of conv features, conv filter height, width, stride in y,x dir.},
      followed by number of GRU layers and number of GRU cells per layer
  Optionally, the bi-directional GRUs and/or GRU with layer-normalization
    can be explored.
  """
  if is_training:
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  # CNN part
  first_filter_count = 198
  first_filter_height = 8
  first_filter_width = 2
  first_filter_stride_y = 3
  first_filter_stride_x = 2

  first_weights = tf.compat.v1.get_variable('W', shape=[first_filter_height,
                    first_filter_width, 1, first_filter_count],
    initializer=tf.keras.initializers.glorot_normal())

  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID')
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)
  else:
    first_dropout = first_relu
  first_conv_output_width = int(math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x))
  first_conv_output_height = int(math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y))

  # GRU part
  num_rnn_layers = 2
  RNN_units = 91
  flow = tf.reshape(first_dropout, [-1, first_conv_output_height,
           first_conv_output_width * first_filter_count])
  cell_fw = []
  for i in range(num_rnn_layers):
    cell_fw.append(tf.keras.layers.GRUCell(RNN_units))

  cells = tf.keras.layers.StackedRNNCells(cell_fw)
  _, last = tf.nn.dynamic_rnn(cell=cells, inputs=flow, dtype=tf.float32)
  flow_dim = RNN_units
  flow = last[-1]

  first_fc_output_channels = 30

  first_fc_weights = tf.compat.v1.get_variable('fcw', shape=[flow_dim,
    first_fc_output_channels],
    initializer=tf.keras.initializers.glorot_normal())
  first_fc = tf.nn.relu(tf.matmul(flow, first_fc_weights))
  if is_training:
    final_fc_input = tf.nn.dropout(first_fc, rate=dropout_rate)
  else:
    final_fc_input = first_fc

  final_fc_weights = tf.Variable(
      tf.random.truncated_normal(
          [first_fc_output_channels, 2], stddev=0.01))

  final_fc = tf.matmul(final_fc_input, final_fc_weights)
  if is_training:
     return final_fc, dropout_rate
  else:
     return final_fc
