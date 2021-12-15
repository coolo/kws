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
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import tensorflow as tf
import glob
import wave
import struct
from python_speech_features import logfbank

FLAGS = None

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_glob, output_layer_name):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)

    list = glob.glob(wav_glob)
    for wav_path in glob.glob(wav_glob):
       w = wave.open(wav_path)
       astr = w.readframes(w.getframerate())
       # convert binary chunks to short 
       a = struct.unpack("%ih" % (w.getframerate()* w.getnchannels()), astr)
       a = [float(val) / pow(2, 15) for val in a]
       wav_data=np.array(a,dtype=float)
       nfft=1024
       mels=logfbank(wav_data, w.getframerate(), lowfreq=50.0, highfreq=4200.0,nfilt=36,winlen=0.020, winstep=0.010, nfft=nfft)
       np.set_printoptions(threshold=np.inf)
       input = {'fingerprint_4d:0': np.reshape(mels, (1, mels.shape[0], mels.shape[1], 1))}
       predictions, = sess.run(softmax_tensor, input)
       print(bcolors.OKGREEN if predictions[1] > predictions[0] else bcolors.FAIL, int(predictions[1] * 100 + 0.5), wav_path, bcolors.ENDC)
    return 0


def label_wav(wav, graph, output_name):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  # load graph, which is stored in the default session
  load_graph(graph)

  run_graph(wav, output_name)


def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.graph, FLAGS.output_name)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
