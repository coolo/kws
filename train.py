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
#
# Modifications Copyright 2017 Arm Inc. All Rights Reserved.           
# Added model dimensions as command line argument and changed to Adam optimizer
#
#

import argparse
import os.path
import sys

import numpy as np
import tensorflow as tf

import logging
import input_data
import models
import tf_slim as slim 

FLAGS = None
SAMPLE_RATE = 16000

def main(_):
  # We want to see all the logging messages for this tutorial.
  logger = tf.get_logger()
  logger.setLevel('INFO')

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(FLAGS.dct_coefficient_count)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_good, FLAGS.data_bad, 
      FLAGS.validation_percentage, model_settings)

  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  inputs = tf.keras.Input(shape=(input_time_size, input_frequency_size, 1), name="fingerprint_4d")
  logits = models.create_model(inputs, model_settings)
  model = tf.keras.Model(inputs=inputs, outputs=logits)
  model.summary()

  # Instantiate an optimizer.
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.007)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
  model.save('saved.model')

  earlystop= tf.keras.callbacks.EarlyStopping(monitor='loss', patience=400, restore_best_weights=True)
  reducelr = tf.keras.callbacks.ReduceLROnPlateau(patience=300, monitor='loss', min_lr=1.0000e-08)
  saver = tf.keras.callbacks.ModelCheckpoint(filepath='saved.model.weighs.{loss:.5f}-{epoch:04d}.h5',  save_weights_only=True, save_best_only=True, monitor='accuracy')
  x_train, y_train = audio_processor.get_data( -1, 0, model_settings, 'training')
  model.fit(x_train, y_train, epochs=10000, batch_size=FLAGS.batch_size, callbacks=[earlystop,reducelr,saver])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_good',
      type=str,
      default='gut',
      help="""\
      Subdirs with good examples.
      """)
  parser.add_argument(
        '--data_bad',
        type=str,
        default='schlecht',
        help="""\
        Subdirs with bad examples.
        """)
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=200,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
