#!/usr/bin/env python3
# Copyright 2017 Google Inc.
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
"""Wraps the audio backend with a simple Python interface for recording and
playback.
"""

import logging
import time
import os

import numpy as np
from python_speech_features import logfbank
from label_wav import calculate_one_sec_mels
import glob
import wave
import struct
import time
import tensorflow as tf

logger = logging.getLogger('audio')

model = tf.keras.models.load_model('saved.model')
model.load_weights('saved.model/best.weights.h5')

winstep = 0.01

for wav in sorted(glob.glob('/mnt/rpi/wavs/*.wav')):
  bt = time.time()
  if os.path.exists(wav + '.npz'):
     data = np.load(wav + '.npz', mmap_mode='r')
     mels_reshaped = data['mels']
  else:
    w = wave.open(wav)
    if w.getnframes() == 0:
      print(wav, 'is incomplete')
      continue
    assert(w.getnchannels() == 1)
    assert(w.getsampwidth() == 2)
    assert(w.getframerate() == 16000)
    frames = w.getnframes()
    astr = w.readframes(frames)
    if len(astr) < frames * w.getnchannels() * 2:
        print(wav, 'is incomplete')
        continue

    # convert binary chunks to short
    a = struct.unpack("%ih" % (frames * w.getnchannels()), astr)
    a = [float(val) / pow(2, 15) for val in a]
    wav_data = np.array(a, dtype=float)
    nfft = 1024
    slices = int((w.getnframes() - w.getframerate() ) / w.getframerate() / winstep)
    mels_reshaped = np.zeros((slices, 99, 36), dtype=np.uint8)

    for slice in range(0, slices):
      start = int(slice*winstep*w.getframerate())
      end = start + w.getframerate()
      slice_data = wav_data[start:end]
      mels = logfbank(slice_data, w.getframerate(), lowfreq=50.0,
                      highfreq=4200.0, nfilt=36, winlen=0.020, winstep=0.010, nfft=nfft)
      mel_clipped = np.uint8((np.clip(mels + 10, -10, 10) / 20 + 0.5) * 256 + 128)
      mels_reshaped[slice] = mel_clipped
    np.savez_compressed(wav + '.npz', mels=mels_reshaped)

  mels_reshaped = np.float32(mels_reshaped)/256
  predictions = np.uint16(model.predict(mels_reshaped)*256+0.5)
  start = 0
  while start < mels_reshaped.shape[0]:
    #print('origin', predictions[start])
    good = predictions[start][1]
    if good > 120:
      print(wav, 'model', start, good)
      w2 = wave.open(wav)
      start_sec = start * winstep
      # skip
      w2.readframes(int(w2.getframerate() * start_sec))
      # read one sec
      astr = w2.readframes(w2.getframerate())
      w2.close()
      new_wav = wave.open(os.path.basename(wav) + '-' + str(start) + '.wav', 'wb')
      new_wav.setparams(w2.getparams())
      new_wav.writeframes(astr)
      new_wav.close()
      start += 50
    else:
      start += 1

  print(wav, 'took', time.time() - bt)
