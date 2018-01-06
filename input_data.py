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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from scipy.io.wavfile import read

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from python_speech_features import logfbank

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59181


def load_wav_file(filename):
  """Loads an audio file and returns a float PCM-encoded array of samples.

  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
  """Saves audio sample data to a .wav audio file.

  Args:
    filename: Path to save the file to.
    wav_data: 2D array of float PCM-encoded audio data.
    sample_rate: Samples per second to encode in the file.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    sample_rate_placeholder = tf.placeholder(tf.int32, [])
    wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
    wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                           sample_rate_placeholder)
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_good, data_bad, silence_percentage, unknown_percentage,
               validation_percentage, testing_percentage,
               model_settings):
    self.data_good = data_good
    self.data_bad = data_bad
    self.prepare_data_index(model_settings, validation_percentage, testing_percentage)
    self.prepare_processing_graph(model_settings)

  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                         filepath)
        tf.logging.error('Please make sure you have enough free space and'
                         ' an internet connection')
        raise
      print()
      statinfo = os.stat(filepath)
      tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                      statinfo.st_size)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_data_index(self, model_settings, validation_percentage, testing_percentage):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
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
        wav_data=np.array(read(wav_path)[1],dtype=float)
        #print(wav_path, len(wav_data))
        #wav_data=np.pad(wav_data, ((0,15000)), 'constant')
        mel=logfbank(wav_data, 16000, lowfreq=125.0,highfreq=3800.0,nfilt=model_settings['dct_coefficient_count'], winlen=model_settings['window_size_ms']/1000,winstep=model_settings['window_stride_ms']/1000)
        #print(mel.shape, model_settings)
        all_files.append({'label': label, 'file': wav_path, 'mels': mel[:model_settings['spectrogram_length']]})
    # Make sure the ordering is random.
    random.shuffle(all_files)

    num_vali = int(len(all_files) * validation_percentage / 100.)
    num_test = int(len(all_files) * testing_percentage / 100.)
    for e in all_files:
       ds = 'training'
       if num_vali > 0:
           ds = 'validation'
           num_vali -= 1
       elif num_test > 0:
           ds = 'testing'
           num_test -= 1
       self.data_index[ds].append(e)

  def prepare_background_data(self):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
      return self.background_data
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
      search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                 '*.wav')
      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        self.background_data.append(wav_data)
      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)

  def prepare_processing_graph(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = model_settings['desired_samples']
    #print(desired_samples)
    self.mels = tf.placeholder(tf.float32, [desired_samples, 1])
    return

    self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)
    # Allow the audio sample's volume to be adjusted.
    self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(wav_decoder.audio,
                                    self.foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    padded_foreground = tf.pad(
        scaled_foreground,
        self.time_shift_padding_placeholder_,
        mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground,
                                 self.time_shift_offset_placeholder_,
                                 [desired_samples, -1])
    # Mix in background noise.
    self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [desired_samples, 1])
    self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
    background_mul = tf.multiply(self.background_data_placeholder_,
                                 self.background_volume_placeholder_)
    background_add = tf.add(background_mul, sliced_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        background_clamp,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    self.mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=model_settings['dct_coefficient_count'])

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
    labels = np.zeros(sample_count)
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
      labels[i] = sample['label']
    return data, labels

  def get_unprocessed_data(self, how_many, model_settings, mode):
    """Retrieve sample data for the given partition, with no transformations.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      model_settings: Information about the current model being trained.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.

    Returns:
      List of sample data for the samples, and list of labels in one-hot form.
    """
    1/0
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many
    desired_samples = model_settings['desired_samples']
    data = np.zeros((sample_count, desired_samples))
    labels = []
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      foreground_volume_placeholder = tf.placeholder(tf.float32, [])
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      foreground_volume_placeholder)
      for i in range(sample_count):
        if how_many == -1:
          sample_index = i
        else:
          sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]
        input_dict = {wav_filename_placeholder: sample['file']}
        if sample['label'] == SILENCE_LABEL:
          input_dict[foreground_volume_placeholder] = 0
        else:
          input_dict[foreground_volume_placeholder] = 1
        data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
        labels.append(sample['label'])
    return data, labels
