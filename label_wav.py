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
import numpy as np
import glob
import os
import wave
import struct
from python_speech_features import logfbank
import pathlib
import hashlib

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

def calculate_one_sec_mels(wav_path, oneseconly=True, after=None, before=None, save_as=None):
    w = wave.open(wav_path)

    assert(w.getnchannels() == 1)
    assert(w.getsampwidth() == 2)
    assert(w.getframerate() == 16000)

    frames = w.getnframes()
    if oneseconly:
        frames = w.getframerate()
    astr = w.readframes(frames)
    if after:
        astr = astr + after
        astr = astr[len(after):]
        if save_as:
            new_wav = wave.open(save_as + '-after.wav', 'wb')
            new_wav.setparams(w.getparams())
            new_wav.writeframes(astr)
            new_wav.close()
    if before:
        astr = before + astr
        astr = astr[0:frames*2]
        if save_as:
            new_wav = wave.open(save_as + '-before.wav', 'wb')
            new_wav.setparams(w.getparams())
            new_wav.writeframes(astr)
            new_wav.close()

    # convert binary chunks to short
    a = struct.unpack("%ih" % (frames * w.getnchannels()), astr)
    a = [float(val) / pow(2, 15) for val in a]
    wav_data = np.array(a, dtype=float)
    nfft = 512
    mels = logfbank(wav_data, w.getframerate(), lowfreq=20.0,
                    nfilt=36, winlen=0.020, winstep=0.010, nfft=nfft,  preemph=0)
    return np.float32(mels)

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
        'a': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32) / 10,
        'b': np.array([2,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32) / 10,
        'c': np.array([4,3,3,3,3,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.float32) / 10,
        'd': np.array([6,6,5,5,3,3,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.float32) / 10,
        'e': np.array([5,3,4,5,6,5,4,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.float32) / 10,
        'f': np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1], dtype=np.float32) / 10,
        'g': np.array([8,8,7,7,5,4,4,3,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.float32) / 10,
        'h': np.array([4,5,6,6,7,6,6,5,4,3,3,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.float32) / 10,
        'i': np.array([8,8,7,7,6,6,5,4,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.float32) / 10,
        'j': np.array([4,3,3,4,5,5,5,5,4,4,4,4,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1], dtype=np.float32) / 10,
        'k': np.array([8,8,7,7,5,4,4,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,1,1], dtype=np.float32) / 10,
        'l': np.array([8,8,7,7,7,7,7,6,5,4,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1], dtype=np.float32) / 10,
        'm': np.array([8,8,7,7,6,6,5,4,4,4,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'n': np.array([4,5,6,7,7,7,7,7,6,5,4,4,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1], dtype=np.float32) / 10,
        'o': np.array([7,7,7,7,6,5,5,4,3,3,3,3,3,3,2,2,2,2,3,3,3,3,3,3,4,3,2,3,3,3,3,3,3,3,2,2], dtype=np.float32) / 10,
        'p': np.array([3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,5,5,5,6,6,6,6,6,6,6,6], dtype=np.float32) / 10,
        'q': np.array([8,8,8,7,7,7,6,5,5,5,4,4,3,3,3,3,3,3,3,3,3,3,2,2,3,2,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'r': np.array([3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,4,3,3,3,3,3,2,2,2,2], dtype=np.float32) / 10,
        's': np.array([8,8,7,7,7,6,5,5,4,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3,2,2,3,3,4,4,5,5,5,5,4], dtype=np.float32) / 10,
        't': np.array([8,8,8,8,8,7,7,7,7,6,5,4,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'u': np.array([8,8,7,7,7,7,7,6,6,5,4,4,3,3,3,3,3,3,3,4,4,4,4,4,4,3,2,2,3,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'v': np.array([3,4,5,6,6,7,7,7,6,6,6,6,6,5,5,5,5,4,4,4,4,4,4,3,3,3,3,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'w': np.array([8,8,8,8,8,7,7,7,6,6,6,5,5,4,4,4,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'x': np.array([7,8,8,8,8,8,8,7,6,6,5,4,3,3,4,5,6,5,4,4,4,4,3,3,3,3,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'y': np.array([7,8,8,8,8,7,7,7,7,7,7,6,6,6,5,5,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'z': np.array([8,8,8,8,7,7,7,6,6,6,5,4,4,4,3,3,3,3,3,3,3,4,4,4,4,3,3,3,3,4,4,4,5,5,4,4], dtype=np.float32) / 10,
        'A': np.array([7,7,7,7,6,6,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,3,3,3,3,3,3,2], dtype=np.float32) / 10,
        'B': np.array([8,8,8,8,8,8,7,7,7,7,6,6,5,5,4,4,3,4,4,5,5,6,5,5,5,4,3,3,3,3,2,2,2,2,2,2], dtype=np.float32) / 10,
        'C': np.array([7,7,7,7,7,6,6,5,5,4,4,4,3,3,3,3,3,3,3,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,5], dtype=np.float32) / 10,
        'D': np.array([8,8,8,8,8,8,8,7,7,7,6,6,6,5,5,5,5,4,4,4,4,4,4,3,4,3,3,3,3,3,3,3,3,3,3,3], dtype=np.float32) / 10,
        'E': np.array([7,8,7,7,8,7,7,7,7,6,6,5,4,4,4,3,3,3,4,4,5,6,6,6,6,5,4,5,5,5,4,4,4,3,3,3], dtype=np.float32) / 10,
        'F': np.array([7,7,8,8,8,8,8,8,7,7,7,7,7,7,7,6,6,6,5,5,5,4,4,4,4,3,3,3,3,2,2,2,2,2,2,2], dtype=np.float32) / 10,
        'G': np.array([8,8,8,8,8,8,8,8,7,7,6,5,5,4,5,6,6,6,6,6,6,5,4,5,5,4,3,3,3,3,3,3,2,2,2,2], dtype=np.float32) / 10,
        'H': np.array([3,4,5,6,6,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,5,5,5,4,4,4,3,3,3,3,3], dtype=np.float32) / 10,
        'I': np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,6], dtype=np.float32) / 10,
        'J': np.array([8,8,8,8,8,8,8,8,7,7,7,7,6,6,6,6,6,5,5,5,5,5,5,4,5,4,4,4,4,3,4,3,3,3,3,3], dtype=np.float32) / 10,
        'K': np.array([8,8,8,8,8,8,7,7,7,7,6,6,5,5,5,5,5,4,5,5,5,5,5,5,5,4,4,4,4,5,5,5,6,6,5,5], dtype=np.float32) / 10,
        'L': np.array([8,8,8,8,8,8,8,8,7,7,7,6,6,6,5,5,5,5,5,6,7,7,6,6,7,6,4,4,4,4,3,3,3,3,3,2], dtype=np.float32) / 10,
        'M': np.array([8,8,8,8,8,8,8,8,8,7,7,7,7,6,6,6,6,6,6,6,6,6,5,5,5,5,4,4,4,4,4,4,4,4,4,4], dtype=np.float32) / 10,
        'N': np.array([7,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,5,4,3,3,3,3,3,3,3,3,3], dtype=np.float32) / 10,
        'O': np.array([7,7,7,7,7,7,7,6,6,6,6,5,5,5,5,5,5,5,5,6,6,6,6,6,7,6,6,7,7,7,7,7,7,7,7,6], dtype=np.float32) / 10,
        'P': np.array([8,8,8,8,8,8,8,7,7,7,7,7,6,6,5,5,5,5,5,6,7,7,7,7,7,7,6,6,6,6,6,5,5,5,4,3], dtype=np.float32) / 10,
        'Q': np.array([7,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,5,4,4,3,3,3,3], dtype=np.float32) / 10,
        'R': np.array([4,4,5,6,7,7,7,7,7,7,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6], dtype=np.float32) / 10,
        'S': np.array([8,8,8,8,8,8,8,8,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,5,5,6,6,6,6,6,6,6,6], dtype=np.float32) / 10,
        'T': np.array([8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,5,5,5,5,5,5,5,5,5,4], dtype=np.float32) / 10,
        'U': np.array([8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6], dtype=np.float32) / 10,
        'V': np.array([8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7], dtype=np.float32) / 10,
        'W': np.array([8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,8,8,8,8,8,8,7,7,7,7,7,6,6,5,5,5,4], dtype=np.float32) / 10,
        'X': np.array([7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,7,7,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7], dtype=np.float32) / 10,
        'Y': np.array([8,8,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7], dtype=np.float32) / 10,
        'Z': np.array([8,8,8,8,9,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8], dtype=np.float32) / 10,
    }
    def difference_between(vec1, vec2):
         return np.linalg.norm(vec1-vec2)

    def short_name(mels):
        # map to -1..-1 for mu encoding
        diagram = np.clip(mels + 7, -10, 10) / 10
        mu = 10
        # map to 0..1
        diagram = (np.sign(diagram) * np.log(1.0 + mu * np.abs(diagram)) / np.log(1.0 + mu) + 1) / 2
        shortname=""
        for x in diagram:
            min=10000
            best_hit=None
            for letter, vec in named_vectors.items():
                diff=difference_between(vec, x)
                if diff < min:
                    min = diff
                    best_hit = letter
            shortname+=best_hit
        # ignore first character - it's too noisy
        return shortname[1:]

    def md5(mels):
        arr = np.uint8((np.clip(mels + 7, -10, 10) / 20 + 0.5) * 256).flatten()
        md5 = hashlib.md5(arr.astype("uint8"))
        return md5.hexdigest()[0:6]

    silence_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'silence.wav')
    w = wave.open(silence_path)
    frames = int(w.getframerate() * 0.08)
    astr = w.readframes(frames)
  
    def predict_wav(wav_path, before=None, after=None, save_as=None):
        mels = calculate_one_sec_mels(wav_path, before=before, after=after, save_as=save_as)

        input_data = np.reshape(mels, (1, mels.shape[0], mels.shape[1]))

        output = interpreter.get_output_details()[0]  # Model has single output.
        input = interpreter.get_input_details()[0]  # Model has single input.
        interpreter.reset_all_variables()
        interpreter.set_tensor(input['index'], input_data)
        interpreter.invoke()

        predictions = np.int8(interpreter.get_tensor(output["index"])[0] * 100 + 0.5)
        return mels, predictions

    for wav_path in sorted(glob.glob(wav_glob)):
        mels, predictions = predict_wav(wav_path)
            
        if FLAGS.rename:
            prefix = "%03d-" % predictions[1]
            if FLAGS.move:
                _, predictions_after = predict_wav(wav_path, after=astr)
                _, predictions_before = predict_wav(wav_path, before=astr)
                prefix += "%03d-" % predictions_before[1]
                prefix += "%03d-" % predictions_after[1]
            sn = prefix + short_name(mels) + "-" + md5(mels) + ".wav"
            if FLAGS.output_moves:
                predict_wav(wav_path, after=astr, save_as=sn)
                predict_wav(wav_path, before=astr, save_as=sn)

            if wav_path == sn:
                print(f"Leave {wav_path}")
                continue
            print(f"rename {wav_path} to {sn}")
            os.rename(wav_path, sn)
        else:
            print(bcolors.OKGREEN if predictions[1] > predictions[0] else bcolors.FAIL,
            predictions[1], wav_path, bcolors.ENDC)

    return 0


def label_wav(wavs, graph):
    """Loads the model and labels, and runs the inference to print predictions."""
    if graph:
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
            interpreter.reset_all_variables()
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
    for wav in wavs:
        run_tflite(wav)


def main(args):
    """Entry point for script, converts flags to arguments."""
    label_wav(args, FLAGS.graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph', type=str, default=None, help='Model to use for identification.')
    parser.add_argument('--rename', dest='rename', action='store_true')
    parser.add_argument('--move', dest='move', action='store_true', help='Check also with 0.1 before and after')
    parser.add_argument('--output_moves', dest='output_moves', action='store_true', help='Output the -before and -after files')
    parser.set_defaults(rename=False)
    parser.set_defaults(move=False)
    parser.set_defaults(output_moves=False)

    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
