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
import tensorflow_model_optimization as tfmot

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


def create_model(model_settings):
    """Builds a model with convolutional recurrent networks with GRUs
    Based on the model definition in https://arxiv.org/abs/1703.05390
    model_size_info: defines the following convolution layer parameters
        {number of conv features, conv filter height, width, stride in y,x dir.},
        followed by number of GRU layers and number of GRU cells per layer
    Optionally, the bi-directional GRUs and/or GRU with layer-normalization
      can be explored.
    """
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    # CNN part
    first_filter_count = 198
    first_filter_height = 8
    first_filter_width = 2
    first_filter_stride_y = 3
    first_filter_stride_x = 2

    inputs = tf.keras.Input(
        shape=(input_time_size, input_frequency_size, 1), name="fingerprint")

    flow = tf.keras.layers.Conv2D(first_filter_count, kernel_size=(first_filter_height, first_filter_width),
                                  strides=(first_filter_stride_y, first_filter_stride_x), padding='valid', activation='relu', name='conv1')(inputs)

    flow = tf.keras.layers.Dropout(0.3)(flow)

    first_conv_output_width = int(math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x))
    first_conv_output_height = int(math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y))

    # GRU part
    num_rnn_layers = 2
    RNN_units = 91
    flow = tf.keras.layers.Reshape(
        (first_conv_output_height, first_conv_output_width * first_filter_count))(flow)
    cells = []
    for _ in range(num_rnn_layers):
        cells.append(tf.keras.layers.GRUCell(RNN_units))

    cells = tf.keras.layers.StackedRNNCells(cells)
    flow = tf.keras.layers.RNN(cells)(flow)

    first_fc_output_channels = 30

    flow = tf.keras.layers.Dense(
        first_fc_output_channels, activation='relu')(flow)
    flow = tf.keras.layers.Dropout(0.1)(flow)

    # Output layer
    flow = tf.keras.layers.Dense(2)(flow)
    flow = tf.keras.layers.Softmax(name='logits')(flow)

    return tf.keras.Model(inputs=inputs, outputs=flow)
