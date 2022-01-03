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
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import wave

from tensorflow.python.platform import gfile

from python_speech_features import logfbank

RANDOM_SEED = 59181

class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_good, data_bad, silence_percentage, unknown_percentage,
               validation_percentage, testing_percentage,
               model_settings):
    self.data_good = data_good
    self.data_bad = data_bad
    self.prepare_data_index(model_settings, validation_percentage, testing_percentage)
    self.prepare_processing_graph(model_settings)

  def prepare_data_index(self, model_settings, validation_percentage, testing_percentage):
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    self.read_one_half(self.data_bad, 0, validation_percentage, testing_percentage, model_settings)
    self.read_one_half(self.data_good, 1, validation_percentage, testing_percentage, model_settings)

  def read_one_half(self, dir, label, validation_percentage, testing_percentage, model_settings):
    search_path = os.path.join(dir, '*.wav')
    all_files = []
    np.set_printoptions(threshold=np.inf)
    for wav_path in gfile.Glob(search_path):
        w = wave.open(wav_path)
        astr = w.readframes(w.getframerate())
        # convert binary chunks to short 
        a = struct.unpack("%ih" % (w.getframerate()* w.getnchannels()), astr)
        a = [float(val) / pow(2, 15) for val in a]
        wav_data=np.array(a,dtype=float)
        #print(wav_path, len(wav_data))
        #wav_data=np.pad(wav_data, ((0,15000)), 'constant')
        nfft=512
        while nfft < model_settings['window_size_samples']:
            nfft *= 2
        mel=logfbank(wav_data, w.getframerate(), lowfreq=50.0,highfreq=4200.0,nfilt=model_settings['dct_coefficient_count'],
                     winlen=model_settings['window_size_ms']/1000,
                     winstep=model_settings['window_stride_ms']/1000,
                     nfft=nfft)
        #print(mel.shape, model_settings)
        all_files.append({'label': label, 'file': wav_path, 'mels': mel[:model_settings['spectrogram_length']]})
        w.close()
    # Make sure the ordering is random.
    random.shuffle(all_files)

    num_vali = int(len(all_files) * validation_percentage / 100.)
    num_test = int(len(all_files) * testing_percentage / 100.)
    for e in all_files:
       self.data_index['validation'].append(e)
       ds = 'training'
       if num_vali > 0:
           ds = 'validation'
           num_vali -= 1
           continue
       elif num_test > 0:
           ds = 'testing'
           num_test -= 1
       self.data_index[ds].append(e)

  def prepare_processing_graph(self, model_settings):
    desired_samples = model_settings['desired_samples']
    self.mels = tf.placeholder(tf.float32, [desired_samples, 1])

  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, model_settings, time_shift, mode, sess):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift: How much to randomly shift the clips by in time.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of label indexes
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    # Data and labels will be populated and returned.
    data = np.zeros((sample_count, model_settings['spectrogram_length'], model_settings['dct_coefficient_count'], 1))
    labels = np.zeros((sample_count,2))
    desired_samples = model_settings['desired_samples']
    pick_deterministically = (mode != 'training')
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(0, sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i + offset
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
      #print(model_settings, sample['mels'].shape)
      # Run the graph to produce the output audio.
      data[i] = np.reshape(sample['mels'][:model_settings['spectrogram_length']], (model_settings['spectrogram_length'], model_settings['dct_coefficient_count'], 1))
      labels[i][sample['label']] = 1
    return data, labels

  