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
r"""Run the LiteRT (TFLite) model against WAVE files and report predictions."""

import argparse
import numpy as np
import glob
import os
import wave
import struct
from python_speech_features import logfbank
import pathlib
import hashlib
from pathlib import Path

try:  # pragma: no cover - runtime dependency
    from ai_edge_litert.interpreter import Interpreter
except ImportError as import_error:  # pragma: no cover
    raise SystemExit(
        "ai_edge_litert is required. Install it via pip install ai-edge-litert"
    ) from import_error

FLAGS = None
SAMPLE_RATE = 16000
DCT_COEFFICIENT_COUNT = 36
WINDOW_SIZE_MS = 0.020
WINDOW_STRIDE_MS = 0.010
NFFT = 512

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
    assert(w.getframerate() == SAMPLE_RATE)

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
    mels = logfbank(
        wav_data,
        w.getframerate(),
        lowfreq=20.0,
        nfilt=DCT_COEFFICIENT_COUNT,
        winlen=WINDOW_SIZE_MS,
        winstep=WINDOW_STRIDE_MS,
        nfft=NFFT,
        preemph=0,
    )
    return np.float32(mels)

def create_interpreter(model_path: Path, num_threads: int):
    interpreter = Interpreter(model_path=str(model_path), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def run_litert(wav_patterns: list[str], model_path: Path, threads: int):
    interpreter = create_interpreter(model_path, threads)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

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
            best_distance = 10000.0
            best_hit=None
            for letter, vec in named_vectors.items():
                diff=difference_between(vec, x)
                if diff < best_distance:
                    best_distance = diff
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
        tensor = np.expand_dims(mels, axis=0).astype(input_details['dtype'])
        reset_fn = getattr(interpreter, "reset_all_variables", None)
        if callable(reset_fn):
            reset_fn()
        interpreter.set_tensor(input_details['index'], tensor)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details['index'])[0].astype(np.float32)
        gut_idx = 1 if len(scores) > 1 else 0
        gut_score = float(scores[gut_idx])
        gut_percent = int(max(0.0, min(100.0, gut_score * 100.0)) + 0.5)
        return mels, gut_percent

    resolved_wavs: list[str] = []
    for pattern in wav_patterns:
        matches = glob.glob(pattern)
        if matches:
            resolved_wavs.extend(matches)
        else:
            resolved_wavs.append(pattern)

    for wav_path in sorted(resolved_wavs):
        mels, gut_percent = predict_wav(wav_path)
            
        if FLAGS.rename:
            prefix = "%03d-" % gut_percent
            if FLAGS.move:
                _, predictions_after = predict_wav(wav_path, after=astr)
                _, predictions_before = predict_wav(wav_path, before=astr)
                prefix += "%03d-" % predictions_before
                prefix += "%03d-" % predictions_after
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
            color = bcolors.OKGREEN if gut_percent >= 50.0 else bcolors.FAIL
            print(
                f"{color}gut={gut_percent:6.2f}% | {wav_path}{bcolors.ENDC}"
            )

    return 0


def main(options):
    if not options.wav_files:
        print("Error: No WAV files specified")
        return 1
    return run_litert(options.wav_files, Path(options.model), options.threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run TFLite inference on WAV files for keyword spotting')
    parser.add_argument(
        'wav_files',
        nargs='+',
        help='WAV files to process (can use glob patterns)')
    parser.add_argument(
        '--rename',
        dest='rename',
        action='store_true',
        help='Rename files based on prediction scores')
    parser.add_argument(
        '--move',
        dest='move',
        action='store_true',
        help='Check also with 0.1s silence before and after')
    parser.add_argument(
        '--output_moves',
        dest='output_moves',
        action='store_true',
        help='Output the -before and -after files')
    parser.add_argument(
        '--model',
        default='model.tflite',
        help='Path to LiteRT/TFLite model to run (default: model.tflite)')
    parser.add_argument(
        '--threads',
        type=int,
        default=2,
        help='Number of CPU threads for LiteRT interpreter')
    parser.set_defaults(rename=False, move=False, output_moves=False)

    args = parser.parse_args()
    FLAGS = args
    
    exit(main(args))
