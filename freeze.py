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
r"""Converts a trained checkpoint into a frozen model for mobile inference.

Once you've trained a model using the `train.py` script, you can use this tool
to convert it into a binary GraphDef file that can be loaded into the Android,
iOS, or Raspberry Pi example code. Here's an example of how to run it:

bazel run tensorflow/examples/speech_commands/freeze -- \
--dct_coefficient_count=40 \
--window_stride_ms=10 \
--model_architecture=conv \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
--output_file=/tmp/my_frozen_graph.pb

The resulting graph has an input for WAV-encoded data named 'wav_data', one for
raw PCM data (as floats in the range -1.0 to 1.0) called 'decoded_sample_data',
and the output is called 'labels_softmax'.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf
import numpy as np

import models
from tensorflow.python.framework import graph_util

FLAGS = None

def main(args):

  tf.compat.v1.disable_eager_execution()

  # Start a new TensorFlow session.
  sess = tf.compat.v1.InteractiveSession()

  model_settings = models.prepare_model_settings(FLAGS.dct_coefficient_count)
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  fingerprint_input = tf.compat.v1.placeholder(
      tf.float32, [None, input_time_size, input_frequency_size, 1], name='fingerprint_4d')

  logits = models.create_model( fingerprint_input, model_settings)

  # Create an output to use for inference.
  output = tf.nn.softmax(logits, name='labels_softmax')

  saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
  saver.restore(sess, args[1])

  for v in tf.compat.v1.trainable_variables():
    var_name = str(v.name)
    var_values = sess.run(v)
    min_value = var_values.min()
    max_value = var_values.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
    dec_bits = 7-int_bits
    # convert to [-128,128) or int8
    var_values = np.round(var_values*2**dec_bits)
    var_name = var_name.replace('/','_')
    var_name = var_name.replace(':','_')
    # convert back original range but quantized to 8-bits or 256 levels
    var_values = var_values/(2**dec_bits)
    # update the weights in tensorflow graph for quantizing the activations
    var_values = sess.run(tf.compat.v1.assign(v,var_values))
    print(var_name+' number of wts/bias: '+str(var_values.shape)+\
            ' dec bits: '+str(dec_bits)+\
            ' max: ('+str(var_values.max())+','+str(max_value)+')'+\
            ' min: ('+str(var_values.min())+','+str(min_value)+')')

  # Turn all the variables into inline constants inside the graph and save it.
  frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, sess.graph_def, ['labels_softmax'])
  tf.io.write_graph(
      frozen_graph_def,
      os.path.dirname(FLAGS.output_file),
      os.path.basename(FLAGS.output_file),
      as_text=False)
  converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=FLAGS.output_file,
    input_arrays=['fingerprint_4d'],
    input_shapes={'fingerprint_4d' : [1, 99,36, 1]},
    output_arrays=['labels_softmax'],
  )
  converter.optimizations = {tf.lite.Optimize.DEFAULT}
  converter.change_concat_input_ranges = True
  tflite_model = converter.convert()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--output_file', type=str, help='Where to save the frozen graph.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
