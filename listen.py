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

import numpy as np
from python_speech_features import logfbank
import audioop
import alsaaudio

try:  # pragma: no cover - runtime dependency
    from ai_edge_litert.interpreter import Interpreter
except ImportError as import_error:  # pragma: no cover
    raise SystemExit(
        "ai_edge_litert is required. Install it via pip install ai-edge-litert"
    ) from import_error

logger = logging.getLogger('audio')

SAMPLE_RATE = 16000
DCT_COEFFICIENT_COUNT = 36
WINDOW_SIZE_MS = 0.020
WINDOW_STRIDE_MS = 0.010
NFFT = 512


def _calculate_mels_from_pcm(pcm: np.ndarray) -> np.ndarray:
    return np.float32(
        logfbank(
            pcm,
            SAMPLE_RATE,
            lowfreq=20.0,
            nfilt=DCT_COEFFICIENT_COUNT,
            winlen=WINDOW_SIZE_MS,
            winstep=WINDOW_STRIDE_MS,
            nfft=NFFT,
            preemph=0,
        )
    )


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
            self.inp = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE, device=input_device)
            self.inp.setchannels(self._channels)
            self.inp.setrate(sample_rate_hz)
            self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)

            # The period size controls the internal number of frames per period.
            # The significance of this parameter is documented in the ALSA api.
            # For our purposes, it is suficcient to know that reads from the device
            # will return this many frames. Each frame being 2 bytes long.
            self.inp.setperiodsize(int(sample_rate_hz * 0.05))
        else:
            url_cmd = ''
            stream = os.environ['STREAM']
            if stream.endswith('.m3u'):
                url_cmd = '-@'
            p = subprocess.Popen('curl -s -L "{}" | mpg123 -m -s -r44100 {} - 2>/dev/null'.format(
                stream, url_cmd), shell=True,  stdout=subprocess.PIPE, close_fds=True)
            self.inp = p.stdout
            self._channels = 1
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
                input_data = self.inp.read(
                    int(self._sample_rate_hz * self._channels * self._bytes_per_sample * 0.1))
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
        required_bytes = int(
            seconds * self._sample_rate_hz * 2 * self._channels)
        if len(self.unscaled_data) < required_bytes:
            return []
        self.lock.acquire()
        passing = self.unscaled_data[-required_bytes:]
        self.unscaled_data = passing
        self.lock.release()
        passing, self._conv_state = audioop.ratecv(
            passing, 2, self._channels, self._sample_rate_hz, 16000, self._conv_state)
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
        self.last_confidences = []

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
            rate = int(self.processor.add_data(chunk) * 25600 + 0.5)
            self.last_confidences.append(rate)
            if len(self.last_confidences) > 5:
                self.last_confidences.pop(0)

            if rate > 1000:
                print('Confidence {} {}'.format(time.time(),
                      self.last_confidences), file=sys.stderr)
                sys.stderr.flush()
                with open('out-%05d-%.2f.raw' % (rate, time.time()), 'wb') as f:
                    f.write(chunk)

            rate = rate / 100
            time_since_last_top = current_time_ms - previous_top_label_time_

            if rate > 3:
                if not 'STREAM' in os.environ and rate >= self.detection_threshold_ and self.last_confidences[-2] >= self.detection_threshold_ and time_since_last_top > self.processor.suppression_ms_:
                    os.system('curl -s http://localhost:3838 &')
                    previous_top_label_time_ = current_time_ms

            delta = time.time() * 1000 - current_time_ms
            if delta < 150:
                time.sleep(.150 - delta / 1000)

            # for p in self._processors:
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

    def __init__(self, model_path, average_window_duration_ms,
                 suppression_ms, minimum_count, sample_rate, sample_duration_ms, num_threads):
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
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.reset_fn = getattr(self.interpreter, "reset_all_variables", None)

    def add_data(self, data_bytes):
        """Process audio data."""
        if not data_bytes:
            return
        #t1 = time.time() * 1000
        input_data = np.frombuffer(data_bytes, dtype='i2')/pow(2, 15)

        mels = _calculate_mels_from_pcm(input_data)
        tensor = np.expand_dims(mels, axis=0).astype(self.input_details['dtype'])
        if callable(self.reset_fn):
            self.reset_fn()
        self.interpreter.set_tensor(self.input_details['index'], tensor)
        self.interpreter.invoke()

        predictions = self.interpreter.get_tensor(self.output_details['index'])[0].astype(np.float32)
        #print('model', time.time() * 1000 - t1, predictions, file=sys.stderr)

        return float(predictions[1] if len(predictions) > 1 else predictions[0])

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
        '--model',
        type=str,
        default='model.tflite',
        help='LiteRT/TFLite model path (default: model.tflite).')
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
    parser.add_argument(
        '--threads',
        type=int,
        default=2,
        help='Number of CPU threads for the LiteRT interpreter.')
    args = parser.parse_args()

    recorder = Recorder(
        input_device=args.input_device,
        bytes_per_sample=args.bytes_per_sample,
        sample_rate_hz=args.rate)

    fetcher = Fetcher(recorder, args.detection_threshold)

    recognizer = RecognizeCommands(
        args.model,
        args.average_window_duration_ms,
        args.suppression_ms,
        args.minimum_count,
        args.sample_rate,
        args.sample_duration_ms,
        args.threads,
    )

    with fetcher, recorder, recognizer:
        fetcher.set_processor(recognizer)
        while not recognizer.is_done():
            time.sleep(0.03)


if __name__ == '__main__':
    main()
