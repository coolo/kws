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
    # Loosely https://arxiv.org/abs/1703.05390
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    # CNN part
    first_filter_count = 198
    first_filter_height = 8
    first_filter_width = 2
    first_filter_stride_y = 3
    first_filter_stride_x = 2

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_time_size, input_frequency_size), name='fingerprint'))
    model.add(tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)))

    conv1 = tf.keras.layers.Conv2D(first_filter_count, kernel_size=(first_filter_height, first_filter_width),
                                      strides=(first_filter_stride_y, first_filter_stride_x), padding='valid', activation='relu', name='conv1')
    model.add(conv1)

    model.add(tf.keras.layers.Dropout(0.3))

    first_conv_output_width = int(math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x))
    first_conv_output_height = int(math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y))

    # GRU part
    RNN_units = 91
    model.add(tf.keras.layers.Reshape(
        (first_conv_output_height, first_conv_output_width * first_filter_count)))
    model.add(tf.keras.layers.LSTM(RNN_units, name='lstm_1', time_major=False, return_sequences=True))

    model.add(tf.keras.layers.Flatten())
    first_fc_output_channels = 30

    dense1 = tf.keras.layers.Dense(first_fc_output_channels, activation='relu', name='dense1')
    model.add(dense1)
    model.add(tf.keras.layers.Dropout(0.1))

    # Output layer
    dense2 = tf.keras.layers.Dense(2, activation=tf.nn.softmax, name='dense2')
    model.add(dense2)
    
    model.add(tfmot.quantization.keras.quantize_annotate_layer(dense2))
    model.add(dense2)

    return tfmot.quantization.keras.quantize_apply(model)
