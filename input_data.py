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
"""Model definitions for simple speech recognition.

"""
import os.path
import random
import struct

import numpy as np
from models import WINDOW_STRIDE_MS  # pylint: disable=redefined-builtin
import tensorflow as tf
import wave

from tensorflow.python.platform import gfile

from python_speech_features import logfbank

RANDOM_SEED = 59181
WINDOW_SIZE_MS = 20
WINDOW_STRIDE_MS = 10


class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, data_good, data_bad, model_settings, old_x, old_ids):
        self.data_good = data_good
        self.data_bad = data_bad
        self.old = dict()
        if len(old_ids):
            for i, file in enumerate(old_ids):
                key = file.decode('utf-8')
                if key in self.old:
                    print('Duplicated key', key)
                self.old[key] = old_x[i]
        self.model_settings = model_settings
        self.prepare_data()

    def prepare_data(self):
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        self.data = []
        self.read_one_half(self.data_bad, 0)
        #self.read_one_half('ignored', 0)
        self.read_one_half(self.data_good, 1)

    def read_one_half(self, dir, label):
        search_path = os.path.join(dir, '*.wav')
        np.set_printoptions(threshold=np.inf)
        for wav_path in gfile.Glob(search_path):
            id = '-'.join(wav_path.split('-')[-2:])
            if id in self.old:
                mels = self.old[id]
            else:
                print(wav_path, 'read')
                w = wave.open(wav_path)
                astr = w.readframes(w.getframerate())
                # convert binary chunks to short
                a = struct.unpack("%ih" %
                                  (w.getframerate() * w.getnchannels()), astr)
                a = [float(val) / pow(2, 15) for val in a]
                wav_data = np.array(a, dtype=float)
                mels = logfbank(wav_data, w.getframerate(), lowfreq=50.0, highfreq=4200.0, nfilt=self.model_settings['dct_coefficient_count'],
                                winlen=WINDOW_SIZE_MS/1000,
                                winstep=WINDOW_STRIDE_MS/1000,
                                nfft=1024)
                # very likely pointless
                mels = mels[:self.model_settings['spectrogram_length']]
                mels = np.reshape(
                    mels, (self.model_settings['spectrogram_length'], self.model_settings['dct_coefficient_count']))
                w.close()
            self.data.append({'label': label, 'mels': mels, 'id': id})

    def get_data(self):
        sample_count = len(self.data)
        # Data and labels will be populated and returned.
        data = np.zeros(
            (sample_count, self.model_settings['spectrogram_length'], self.model_settings['dct_coefficient_count']))
        labels = np.zeros((sample_count, 2))
        ids = np.zeros(sample_count, dtype='|S120')
        candidates = list(range(0, sample_count))
        random.shuffle(candidates)

        for i in range(0, sample_count):
            sample = self.data[candidates[i]]
            data[i] = sample['mels']
            labels[i][sample['label']] = 1
            ids[i] = sample['id']
        return data, labels, ids


def get_data(data_good, data_bad, model_settings, old_x, old_ids):
    ap = AudioProcessor(data_good, data_bad, model_settings, old_x, old_ids)
    return ap.get_data()
