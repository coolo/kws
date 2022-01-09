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
import logging
import os.path
import sys

import numpy as np
import tensorflow as tf

import input_data
import models

FLAGS = None
SAMPLE_RATE = 16000


def main(_):
    # We want to see all the logging messages for this tutorial.
    logger = tf.get_logger()
    logger.setLevel('INFO')

    model_settings = models.prepare_model_settings(FLAGS.dct_coefficient_count)
    model = models.create_model(model_settings)
    model.summary()

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00007)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=["accuracy"])
    model.save('saved.model')

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=12, restore_best_weights=True)
    saver = tf.keras.callbacks.ModelCheckpoint(
        filepath='saved.model.weighs.{epoch:04d}.h5',  save_weights_only=True, save_best_only=True, monitor='accuracy')
    if FLAGS.rescan:
        audio_processor = input_data.AudioProcessor(
            FLAGS.data_good, FLAGS.data_bad,
            FLAGS.validation_percentage, model_settings)

        x_train, y_train = audio_processor.get_data(model_settings)
        np.savez('all-waves.npz', x=x_train, y=y_train)
    else:
        data = np.load('all-waves.npz', mmap_mode='r')
        x_train = data['x']
        y_train = data['y']
    model.fit(x_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, callbacks=[
              earlystop, saver], validation_split=0.05)


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
        '--rescan',
        type=bool,
        default=False,
        help="Rescan the wav files")
    parser.add_argument(
        '--epochs',
        type=int,
        default=9999,
        help="Epochs to run max")
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=36,
        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',)

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
