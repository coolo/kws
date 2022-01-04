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

  # Start a new TensorFlow session.
  sess = tf.compat.v1.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(FLAGS.dct_coefficient_count)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_good, FLAGS.data_bad, 
      FLAGS.validation_percentage, model_settings)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  fingerprint_input = tf.compat.v1.placeholder(
      tf.float32, [None, input_time_size, input_frequency_size, 1], name='fingerprint_4d')

  logits, dropout_rate = models.create_model(
      fingerprint_input,
      model_settings,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.compat.v1.placeholder(
      tf.float32, [None, 2], name='groundtruth_input')
  
  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=ground_truth_input, logits=logits))
  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)
  
  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  with tf.name_scope('train'), tf.control_dependencies(update_ops):
    learning_rate_input = tf.compat.v1.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate_input)
    train_step = slim.learning.create_train_op(cross_entropy_mean, train_op)
  predicted_indices = tf.argmax(logits, 1)
  expected_indices = tf.argmax(ground_truth_input, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.math.confusion_matrix(
      expected_indices, predicted_indices, num_classes=2)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  global_step = tf.compat.v1.train.get_or_create_global_step()
  increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)

  saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.compat.v1.summary.merge_all()
 
  tf.compat.v1.global_variables_initializer().run()

  # Parameter counts
  params = tf.compat.v1.trainable_variables()
  num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
  print('Total number of Parameters: ', num_params)

  start_step = 1
  logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.io.write_graph(sess.graph_def, FLAGS.train_dir,
                       'crnn.pbtxt')

  # Training loop.
  best_accuracy = 0
  training_steps_max = np.sum(training_steps_list)
  for training_step in range(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, 'training')
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_rate: 0
        })
    logging.info('Step #%d: rate %f, accuracy %.2f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in range(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 'validation'))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_rate: 0
            })
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      logging.info('Step %d: Validation accuracy = %.2f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))

      # Save the model checkpoint when validation accuracy improves
      if total_accuracy >= best_accuracy:
        best_accuracy = total_accuracy
        checkpoint_path = os.path.join(FLAGS.train_dir, 'best',
                                       'crnn_'+ str(int(best_accuracy*10000)) + '.ckpt')
        logging.info('Saving best model to "%s-%d"', checkpoint_path, training_step)
        saver.save(sess, checkpoint_path, global_step=training_step)
      logging.info('So far the best validation accuracy is %.2f%%' % (best_accuracy*100))

    


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
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
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
