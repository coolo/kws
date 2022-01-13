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

from numpy.core.shape_base import block
import tensorflow as tf
import argparse
import numpy as np
import glob
import os
import wave
import struct
from python_speech_features import logfbank
import pathlib

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

    model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'model.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()  # Needed before execution!

    np.set_printoptions(threshold=np.inf, linewidth=1000)

    named_vectors = {
        'a': np.array([5,6,6,7,7,7,7,7,7,7,7,6,6,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,5,5,5,6,6,5,4,4], dtype=np.float32) / 10,
        'b': np.array([4,4,3,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,5,5,4,4,4], dtype=np.float32) / 10,
        'c': np.array([4,4,4,4,4,5,7,7,5,4,5,6,7,7,6,6,7,7,6,6,6,6,6,6,7,7,8,8,8,8,8,8,8,7,7,7], dtype=np.float32) / 10,
        'd': np.array([4,4,3,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'e': np.array([3,4,4,5,5,5,6,6,6,6,6,4,4,3,3,3,3,3,3,4,5,6,6,5,5,5,6,6,4,4,4,4,5,4,3,3], dtype=np.float32) / 10,
        'f': np.array([4,4,4,5,6,6,6,6,6,7,7,6,6,5,5,5,4,4,4,4,4,4,4,5,6,6,7,7,7,7,7,7,7,7,6,6], dtype=np.float32) / 10,
        'g': np.array([4,5,5,6,6,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,7,7,7,8,8,8,8,8,8,8,7,7,7], dtype=np.float32) / 10,
        'h': np.array([3,3,3,3,3,3,3,4,4,4,4,4,5,5,5,4,4,4,4,3,3,3,3,3,4,5,5,6,6,6,6,6,6,5,5,5], dtype=np.float32) / 10,
        'i': np.array([4,4,3,4,4,4,4,5,5,5,5,5,5,4,4,4,3,3,3,3,3,3,4,4,5,6,6,5,5,5,5,6,6,4,4,4], dtype=np.float32) / 10,
        'j': np.array([5,6,6,7,7,7,7,8,8,8,8,7,7,6,6,6,6,6,6,7,7,8,7,7,7,7,8,7,7,7,7,7,7,7,6,6], dtype=np.float32) / 10,
        'k': np.array([4,4,5,5,6,5,5,5,5,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,3,3,3,3,4,4,3,3,3], dtype=np.float32) / 10,
        'l': np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'm': np.array([4,4,4,4,4,4,4,4,5,5,5,5,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], dtype=np.float32) / 10,
        'n': np.array([3,3,3,3,3,3,4,4,5,5,5,3,3,3,3,3,3,3,3,3,4,4,4,3,3,4,4,4,3,3,4,4,4,3,3,3], dtype=np.float32) / 10,
        'o': np.array([3,3,3,4,4,4,4,5,5,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,4,4,5,5,5,5,5,6,5,4,4,4], dtype=np.float32) / 10,
        'p': np.array([4,5,5,6,6,6,6,6,5,5,4,4,4,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,4,4,4,5,5,4,3,3], dtype=np.float32) / 10,
        'q': np.array([4,5,6,6,6,6,7,7,7,6,6,5,5,4,4,4,4,4,4,5,6,5,5,4,4,5,5,5,5,5,6,6,7,5,4,4], dtype=np.float32) / 10,
        'r': np.array([4,4,4,4,4,4,4,5,6,6,6,6,6,5,5,4,4,4,4,3,3,3,4,4,5,6,7,7,7,6,7,7,7,6,5,6], dtype=np.float32) / 10,
        's': np.array([5,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7], dtype=np.float32) / 10,
        't': np.array([4,4,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,6,6,5,6,6,7,7,7,7,7,7,7,7,6,6,7], dtype=np.float32) / 10,
        'u': np.array([4,4,5,6,6,5,5,6,6,6,6,6,6,5,5,5,4,4,4,4,4,4,4,5,6,6,6,6,5,6,6,6,6,5,4,5], dtype=np.float32) / 10,
        'v': np.array([4,4,4,4,4,6,5,4,4,5,6,7,6,5,5,5,4,4,4,4,3,3,4,4,4,5,5,6,6,7,7,7,7,6,6,6], dtype=np.float32) / 10,
        'w': np.array([4,4,3,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3], dtype=np.float32) / 10,
        'x': np.array([4,4,4,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,6], dtype=np.float32) / 10,
        'y': np.array([3,3,3,3,4,4,6,6,4,4,4,5,6,6,5,4,5,6,5,4,4,4,4,4,5,6,6,7,7,7,7,7,7,7,6,6], dtype=np.float32) / 10,
        'z': np.array([4,5,5,6,6,6,7,7,7,7,6,6,5,4,4,4,4,4,4,5,6,6,5,5,5,5,6,5,4,4,4,4,5,4,4,3], dtype=np.float32) / 10,
        'A': np.array([3,3,3,4,4,5,5,6,6,6,5,4,3,3,3,3,3,3,3,4,5,6,5,4,4,4,5,4,3,3,3,4,4,3,3,3], dtype=np.float32) / 10,
        'B': np.array([4,4,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,2,2,3], dtype=np.float32) / 10,
        'C': np.array([5,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8], dtype=np.float32) / 10,
        'D': np.array([4,4,3,4,5,5,4,4,4,6,6,5,5,6,6,5,5,5,4,4,4,4,4,5,6,6,7,7,7,7,7,7,7,6,6,6], dtype=np.float32) / 10,
        'E': np.array([4,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,6,6,7,7,7,7,7,7,7,7,7,7], dtype=np.float32) / 10,
        'F': np.array([4,4,4,4,5,5,6,6,7,7,7,6,6,7,7,7,7,7,7,7,7,6,7,7,7,8,8,8,8,8,8,8,8,7,7,7], dtype=np.float32) / 10,
        'G': np.array([4,4,4,4,4,5,6,6,6,6,7,7,6,6,6,5,5,5,5,5,5,5,5,6,7,7,7,7,7,7,7,7,7,6,6,6], dtype=np.float32) / 10,
        'H': np.array([5,6,6,7,7,7,7,8,7,7,7,7,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,6,6,6,6,6,5,4,4], dtype=np.float32) / 10,
        'I': np.array([4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,3,3,4], dtype=np.float32) / 10,
        'J': np.array([4,5,5,6,6,6,7,7,7,6,6,5,4,4,4,4,4,4,4,4,5,6,6,6,6,6,7,6,6,5,6,7,7,6,4,4], dtype=np.float32) / 10,
        'K': np.array([4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,2,2,3,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], dtype=np.float32) / 10,
        'L': np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,5,5,5,6,6,6,6,6,5,5,5], dtype=np.float32) / 10,
        'M': np.array([2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,3], dtype=np.float32) / 10,
        'N': np.array([4,4,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,5,5,5,5,6,5,5,4,5], dtype=np.float32) / 10,
        'O': np.array([3,4,4,5,5,4,5,6,6,7,7,5,5,4,4,4,3,4,4,5,6,6,6,5,5,6,6,5,5,5,6,7,7,6,4,4], dtype=np.float32) / 10,
        'P': np.array([4,4,4,4,5,5,5,6,6,6,6,6,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4], dtype=np.float32) / 10,
        'Q': np.array([4,4,4,5,5,6,6,5,4,4,4,5,4,4,5,5,5,5,5,5,5,5,5,6,6,7,7,7,6,6,6,6,6,5,5,6], dtype=np.float32) / 10,
        'R': np.array([4,4,4,4,5,6,6,5,4,5,6,7,7,6,6,7,7,6,6,6,6,6,6,7,7,7,8,8,8,8,8,8,8,7,7,7], dtype=np.float32) / 10,
        'S': np.array([4,4,3,4,3,3,4,4,3,4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3], dtype=np.float32) / 10,
        'T': np.array([4,5,5,5,6,6,6,6,6,6,6,6,6,5,6,6,6,6,6,6,5,5,5,5,5,5,5,4,4,4,4,5,5,4,4,4], dtype=np.float32) / 10,
        'U': np.array([4,4,4,4,4,5,6,6,6,6,6,5,5,4,4,4,4,4,4,4,4,5,5,6,6,6,6,6,6,6,6,6,6,5,5,6], dtype=np.float32) / 10,
        'V': np.array([4,4,5,6,6,6,7,7,7,7,7,6,6,5,5,5,5,5,6,6,7,7,7,6,6,7,7,7,6,6,6,7,7,6,5,4], dtype=np.float32) / 10,
        'W': np.array([4,4,4,5,5,5,5,5,5,5,4,4,4,4,4,4,3,4,3,3,4,4,4,4,4,5,6,6,6,6,6,6,6,6,5,6], dtype=np.float32) / 10,
        'X': np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'Y': np.array([4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], dtype=np.float32) / 10,
        'Z': np.array([4,4,4,4,4,4,5,6,6,6,5,4,3,3,3,3,3,3,3,4,5,5,4,4,4,4,5,4,4,4,5,6,6,4,4,3], dtype=np.float32) / 10
    }
    def difference_between(vec1, vec2):
         return np.linalg.norm(vec1-vec2)

    def short_name(mels):
        # map to -1..-1 for mu encoding
        diagram = (np.clip(mels, -16, 0) + 8) / 16
        mu = 10
        diagram = np.uint8((np.sign(diagram) * (np.log(1.0 + mu * np.abs(diagram)) / np.log(1.0 + mu)) + 1) * 5 + 0.5)
        shortname=""
        for x in diagram:
            min=10000
            best_hit=None
            for letter, vec in named_vectors.items():
                diff=difference_between(vec, np.float32(x) / 10)
                if diff < min:
                    min = diff
                    best_hit = letter
            shortname+=best_hit
        # ignore first character - it's too noisy
        return shortname[1:]

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
        
        input_data = np.float32(np.reshape(mels, (1, mels.shape[0], mels.shape[1])))

        output = interpreter.get_output_details()[0]  # Model has single output.
        input = interpreter.get_input_details()[0]  # Model has single input.
        interpreter.reset_all_variables()
        interpreter.set_tensor(input['index'], input_data)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output["index"])[0] * 255
        if FLAGS.rename:
            sn = "%03d-" % int(predictions[1] * 100 / 255 + 0.5) + short_name(mels)
            counter = 0
            if os.path.exists(sn + ".wav"):
                counter = 1
                while os.path.exists(f"{sn}-{counter}.wav"):
                    counter += 1
                sn = sn + f"-{counter}"
            print(f"rename {wav_path} to {sn}.wav")
            os.rename(wav_path, sn + ".wav")
        else:
            print(bcolors.OKGREEN if predictions[1] > predictions[0] else bcolors.FAIL, int(
            predictions[1] * 100 / 255 + 0.5), wav_path, bcolors.ENDC)

    return 0


def label_wav(wav, graph):
    """Loads the model and labels, and runs the inference to print predictions."""
    if FLAGS.graph:
        model = tf.keras.models.load_model('saved.model')
        model.load_weights(graph)
        # fixed batch size
        model.input.set_shape((1,) + model.input.shape[1:])
        model.summary()

        data = np.load('all-waves.npz', mmap_mode='r')
        dataset = tf.data.Dataset.from_tensor_slices(data['x'])

        def representative_dataset():
            for data in dataset.shuffle().batch(1).take(300):
                yield [tf.dtypes.cast(data, tf.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.representative_dataset = representative_dataset
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]

        tflite_model = converter.convert()
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)

        model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'model.tflite')
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()  # Needed before execution!
	
        dataset = tf.data.Dataset.from_tensor_slices((data['x'], data['y']))
        misses_lite = 0
        misses_model = 0
        count = 0
        for element in dataset.take(1000):
            mels, y = element
            input_data = np.float32(np.reshape(mels, (1, mels.shape[0], mels.shape[1])))
            output = interpreter.get_output_details()[0]  # Model has single output.
            input = interpreter.get_input_details()[0]  # Model has single input.
            interpreter.set_tensor(input['index'], input_data)
            interpreter.invoke()

            true_value = int(y.numpy()[1] * 256)
            predictions_lite = int(interpreter.get_tensor(output["index"])[0][1] * 256 + 0.5)
            predictions_model = int(model(input_data).numpy()[0][1] * 256 + 0.5)
            print("true:", true_value, "lite:", bcolors.OKGREEN if predictions_lite == true_value else bcolors.FAIL, predictions_lite, bcolors.ENDC, "model:", predictions_model)
            count += 1
            if abs(true_value - predictions_lite) > 70:
                misses_lite += 1
            if abs(true_value - predictions_model) > 70:
                misses_model += 1

        print("Accuracy Model:", 100 - float(misses_model) / count, "Lite model:", 100 - float(misses_lite) / count)
    run_tflite(wav)


def main(_):
    """Entry point for script, converts flags to arguments."""
    label_wav(FLAGS.wav, FLAGS.graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav', type=str, default='', help='Audio file to be identified.')
    parser.add_argument(
        '--graph', type=str, default=None, help='Model to use for identification.')
    parser.add_argument('--rename', dest='rename', action='store_true')
    parser.set_defaults(rename=False)

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    main(unparsed)
