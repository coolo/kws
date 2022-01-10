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

import tensorflow as tf
import argparse
import time
import numpy as np
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


def run_tflite(wav_glob):
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class

    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()  # Needed before execution!

    for wav_path in sorted(glob.glob(wav_glob)):
        w = wave.open(wav_path)
        astr = w.readframes(w.getframerate())
        # convert binary chunks to short
        a = struct.unpack("%ih" % (w.getframerate() * w.getnchannels()), astr)
        a = [float(val) / pow(2, 15) for val in a]
        wav_data = np.array(a, dtype=float)
        nfft = 1024
        mels = logfbank(wav_data, w.getframerate(), lowfreq=50.0,
                        highfreq=4200.0, nfilt=36, winlen=0.020, winstep=0.010, nfft=nfft)
        np.set_printoptions(threshold=np.inf)
        input_data = np.float32(np.reshape(
            mels, (1, mels.shape[0], mels.shape[1])))

        output = interpreter.get_output_details()[0]  # Model has single output.
        input = interpreter.get_input_details()[0]  # Model has single input.
        interpreter.set_tensor(input['index'], input_data)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output["index"])[0]
        #print(predictions)
        print(bcolors.OKGREEN if predictions[1] > predictions[0] else bcolors.FAIL, int(
            predictions[1] * 100 + 0.5), wav_path, bcolors.ENDC)
    return 0


def label_wav(wav, graph):
    """Loads the model and labels, and runs the inference to print predictions."""
    if True:
        model = tf.keras.models.load_model('saved.model')
        model.load_weights(graph)
        # fixed batch size
        model.input.set_shape((1,) + model.input.shape[1:])
        model.summary()

        data = np.load('all-waves.npz', mmap_mode='r')
        dataset = tf.data.Dataset.from_tensor_slices(data['x'])

        def representative_dataset():
            for data in dataset.batch(1).take(300):
                yield [tf.dtypes.cast(data, tf.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = representative_dataset
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter._experimental_lower_tensor_list_ops = True
        #converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        #converter.experimental_new_converter = True

        tflite_model = converter.convert()
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
    run_tflite(wav)


def main(_):
    """Entry point for script, converts flags to arguments."""
    label_wav(FLAGS.wav, FLAGS.graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav', type=str, default='', help='Audio file to be identified.')
    parser.add_argument(
        '--graph', type=str, default='', help='Model to use for identification.')

    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
