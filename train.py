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
import sys

import numpy as np
import tensorflow as tf

import input_data
import models

FLAGS = None
SAMPLE_RATE = 16000

class ConfusionMatrixDisplay(tf.keras.callbacks.Callback):
    def __init__(self, X_val, Y_val):
        self.X_val = X_val
        self.Y_val = Y_val

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X_val)
        max_pred = np.argmax(pred, axis=1)
        max_y = np.argmax(self.Y_val, axis=1)
        print()
        print(tf.math.confusion_matrix(max_y, max_pred).numpy())

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
        monitor='loss', patience=FLAGS.epochs, restore_best_weights=True)
    saver = tf.keras.callbacks.ModelCheckpoint(
        filepath='saved.model.weighs.{epoch:04d}.h5',  save_weights_only=True, save_best_only=True, monitor='accuracy')
    if FLAGS.rescan:
        audio_processor = input_data.AudioProcessor(
            FLAGS.data_good, FLAGS.data_bad, model_settings)

        x_train, y_train = audio_processor.get_data(model_settings)
        np.savez('all-waves.npz', x=x_train, y=y_train)
    else:
        data = np.load('all-waves.npz', mmap_mode='r')
        x_train = data['x']
        y_train = data['y']
    plotter2 = ConfusionMatrixDisplay(X_val=x_train, Y_val=y_train)

    model.fit(x_train, y_train, epochs=400, batch_size=100, callbacks=[earlystop, plotter2, saver])
    # evaluate the model
    scores = model.evaluate(x_train, y_train)
    input('Press ENTER to continue...')


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
        default=12,
        help="Epochs to be patient for")
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
