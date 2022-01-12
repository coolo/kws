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

from collections import deque
import logging
import os
import subprocess
import threading
import time
import sys
try:
  import tflite_runtime.interpreter as tflite
except:
  from tensorflow import lite as tflite

import numpy as np
import subprocess
from python_speech_features import logfbank
import audioop
import alsaaudio

logger = logging.getLogger('audio')

def sample_width_to_string(sample_width):
  """Convert sample width (bytes) to ALSA format string."""
  return {1: 's8', 2: 's16', 4: 's32'}[sample_width]

class Recorder(threading.Thread):
  """Stream audio from microphone in a background thread and run processing
    callbacks. It reads audio in a configurable format from the microphone,
    then converts it to a known format before passing it to the processors.
    """
  CHUNK_S = 0.25

  def __init__(self,
               input_device='default',
                bytes_per_sample=2,
               sample_rate_hz=48000):
    """Create a Recorder with the given audio format.

        The Recorder will not start until start() is called. start() is called
        automatically if the Recorder is used in a `with`-statement.

        - bytes_per_sample: sample width in bytes (eg 2 for 16-bit audio)
        - sample_rate_hz: sample rate in hertz
        """

    super(Recorder, self).__init__()

    self._channels = 1
    self._sample_rate_hz = sample_rate_hz
    self._bytes_per_sample = bytes_per_sample
    if not 'STREAM' in os.environ:
       self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, device=input_device)
       self.inp.setchannels(self._channels)
       self.inp.setrate(sample_rate_hz)
       self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)

       # The period size controls the internal number of frames per period.
       # The significance of this parameter is documented in the ALSA api.
       # For our purposes, it is suficcient to know that reads from the device
       # will return this many frames. Each frame being 2 bytes long.
       self.inp.setperiodsize(int(sample_rate_hz * 0.05))
    else:
       p = subprocess.Popen('curl -s -L "{}" | mpg123 -s -@ -'.format(os.environ['STREAM']), shell=True,  stdout=subprocess.PIPE, close_fds=True)
       self.inp = p.stdout
       self._channels = 2
       self._sample_rate_hz = 44100

    self._conv_state = None
    self.lock = threading.Lock()

    self.unscaled_data = b''

  def run(self):
    """Reads data from arecord and passes to processors."""

    logger.info('{} started recording'.format(time.time()))

    while True:
      if not 'STREAM' in os.environ:
          l, input_data = self.inp.read()
      else:
          input_data = self.inp.read(int(self._sample_rate_hz * self._channels * self._bytes_per_sample * 0.1))
      if not input_data:
        break

      self.lock.acquire()
      self.unscaled_data += input_data
      self.lock.release()

    if True:
      logger.error('Microphone recorder died unexpectedly, aborting...')
      # sys.exit doesn't work from background threads, so use os._exit as
      # an emergency measure.
      logging.shutdown()
      os._exit(1)  # pylint: disable=protected-access

  def take_last_chunk(self, seconds):
    required_bytes = int(seconds * self._sample_rate_hz * 2 * self._channels)
    if len(self.unscaled_data) < required_bytes:
        return []
    self.lock.acquire()
    passing = self.unscaled_data[-required_bytes:]
    self.unscaled_data = passing
    self.lock.release()
    passing, self._conv_state = audioop.ratecv(passing, 2, self._channels, self._sample_rate_hz, 16000, self._conv_state)
    if self._channels > 1:
       passing = audioop.tomono(passing, 2, .5, .5)
    return passing

  def __exit__(self, *args):
    pass

  def __enter__(self):
    self.start()
    return self

class Fetcher(threading.Thread):
  def __init__(self, recorder, detection_threshold):
    super(Fetcher, self).__init__()
    self.processor = None
    self.detection_threshold_ = detection_threshold
    self.recorder = recorder

  def __exit__(self, *args):
    pass

  def __enter__(self):
    self.start()
    return self

  def set_processor(self, processor):
    self.processor = processor

  def run(self):
    previous_top_label_time_ = 0
    while True:
      chunk = self.recorder.take_last_chunk(1)
      if not len(chunk):
         time.sleep(0.1)
         continue

      current_time_ms = time.time() * 1000
      rate = self.processor.add_data(chunk)

      if rate>=0:
         print('{} Confidence {:4}'.format(time.time(), int(rate)), file=sys.stderr)
         sys.stderr.flush()

      time_since_last_top = current_time_ms - previous_top_label_time_

      if rate > 0:
        if not 'STREAM' in os.environ and rate >= self.detection_threshold_ and time_since_last_top > self.processor.suppression_ms_:
          os.system('curl -s http://localhost:3838 &')
          previous_top_label_time_ = current_time_ms
        with open('out-%03d-%.2f.raw' % (rate, time.time()), 'wb') as f:
          f.write(chunk)
      
      delta = time.time() * 1000 - current_time_ms
      if delta < 150:
         time.sleep(.150 - delta / 1000)

      #for p in self._processors:
      # return p.add_data(chunk)

class RecognizePredictions(object):

  def __init__(self, time, predictions):
    self.time_ = time
    self.predictions_ = predictions

  def time(self):
    return self.time_

  def predictions(self):
    return self.predictions_

class RecognizeCommands(object):
  """A processor that identifies spoken commands from the stream."""

  def __init__(self, graph, output_name, average_window_duration_ms, 
               suppression_ms, minimum_count, sample_rate, sample_duration_ms):
    self.output_name_ = output_name
    self.average_window_duration_ms_ = average_window_duration_ms
    self.suppression_ms_ = suppression_ms
    self.last_time_ms = 0
    self.minimum_count_ = minimum_count
    self.sample_rate_ = sample_rate
    self.sample_duration_ms_ = sample_duration_ms
    self.previous_top_label_ = '_silence_'
    self.previous_top_label_time_ = 0
    self.previous_top_score = 0
    self.recording_length_ = int((sample_rate * sample_duration_ms) / 1000)
    self.recording_buffer_ = np.zeros(
        [self.recording_length_], dtype=np.float32)
    self.recording_offset_ = 0
    self.interpreter = tflite.Interpreter(model_path='model.tflite')
    self.interpreter.allocate_tensors()
    self.output_tensor = self.interpreter.get_output_details()[0]['index']
    self.input_tensor = self.interpreter.get_input_details()[0]['index']
    self.previous_results_ = deque()

  def add_data(self, data_bytes):
    """Process audio data."""
    if not data_bytes:
      return
    #t1 = time.time() * 1000
    input_data = np.frombuffer(data_bytes, dtype='i2')/pow(2,15)

    mels=logfbank(input_data, 16000, lowfreq=50.0, highfreq=4200.0,nfilt=36,nfft=1024, winlen=0.020,winstep=0.010)
    
    self.interpreter.set_tensor(self.input_tensor, np.float32([mels]))
    self.interpreter.invoke()

    predictions = self.interpreter.get_tensor(self.output_tensor)[0]
    #print('model', time.time() * 1000 - bt * 1000, file=sys.stderr)

    return predictions[1]

  def is_done(self):
    return False

  def __enter__(self):
    return self

  def __exit__(self, *args):
    pass

def main():
  logging.basicConfig(level=logging.INFO)

  import argparse
  import time

  parser = argparse.ArgumentParser(description='Test audio wrapper')
  parser.add_argument(
      '-I',
      '--input-device',
      default='default',
      help='Name of the audio input device')
  parser.add_argument(
      '-f',
      '--bytes-per-sample',
      type=int,
      default=2,
      help='Sample width in bytes')
  parser.add_argument(
      '-r', '--rate', type=int, default=48000, help='Sample rate in Hertz')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--average_window_duration_ms',
      type=int,
      default='300',
      help='How long to average results over.')
  parser.add_argument(
      '--detection_threshold',
      type=float,
      default=0.9,
      help='Score required to trigger recognition.')
  parser.add_argument(
      '--suppression_ms',
      type=int,
      default='800',
      help='How long to ignore recognitions after one has triggered.')
  parser.add_argument(
      '--minimum_count',
      type=int,
      default='2',
      help='How many recognitions must be present in a window to trigger.')
  parser.add_argument(
      '--sample_rate', type=int, default='16000', help='Audio sample rate.')
  parser.add_argument(
      '--sample_duration_ms',
      type=int,
      default='1000',
      help='How much audio the recognition model looks at.')
  args = parser.parse_args()

  recorder = Recorder(
      input_device=args.input_device,
      bytes_per_sample=args.bytes_per_sample,
      sample_rate_hz=args.rate)

  fetcher = Fetcher(recorder, args.detection_threshold)

  recognizer = RecognizeCommands(
      args.graph, args.output_name, args.average_window_duration_ms,
      args.suppression_ms, args.minimum_count,
      args.sample_rate, args.sample_duration_ms)

  with fetcher, recorder, recognizer:
    fetcher.set_processor(recognizer)
    while not recognizer.is_done():
      time.sleep(0.03)

if __name__ == '__main__':
  main()
