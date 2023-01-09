# Copyright 2023 The FACEGOOD Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import sys
import tensorflow as tf
import numpy as np

import time
import logging

from model_paper import losses, Audio2Face

# tf.config.run_functions_eagerly(True) # Run eagerly
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Allow GPU memory growth


def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger


def train(epochs,
          ckpt_epoch,
          model,
          checkpoint_save_path,
          loss_object,
          optimizer,
          train_ds,
          test_ds,
          test_freq=5,
          save_freq=50,
          logger=None):
    """Train the model
    Args:
        epochs: int, the number of epochs to train the model
        ckpt_epoch: int, the number of epochs to train the model
        model: tf.keras.Model, the model to train
        checkpoint_save_path: str, the path to save the checkpoint
        loss_object: tf.keras.losses, the loss function
        optimizer: tf.keras.optimizers, the optimizer
        train_ds: tf.data.Dataset, the training dataset
        test_ds: tf.data.Dataset, the test dataset
        test_freq: int, the frequency to test the model
        save_freq: int, the frequency to save the model
        logger: python logger
    """

    # Loss metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_mse = tf.keras.metrics.MeanSquaredError(name='train_mse')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_mse = tf.keras.metrics.MeanSquaredError(name='test_mse')

    # Restore the latest checkpoint if exists
    if os.path.exists(checkpoint_save_path + ".index") and ckpt_epoch>1 :
        logger.info(f'------------- load the epoch-{ckpt_epoch} model------------')
        model.load_weights(checkpoint_save_path)

    # Train the model for epochs
    for epoch in range(ckpt_epoch+1, epochs + 1):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_mse.reset_states()

        time_start = time.time() # Record the start time of each epoch
        for train_data, labels in train_ds:
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions, emotion_input = model(train_data, training=True)
                loss = loss_object(labels, (predictions, emotion_input))

            gradients = tape.gradient(loss, model.trainable_variables)  # Calculate gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Update weights
            train_loss.update_state(loss)   # Update training metric
            train_mse.update_state(labels, predictions) 
        time_end = time.time()  
        logger.info(    # Log the training information
            f'Training Epoch-{epoch} '
            f'LR: {optimizer._decayed_lr(tf.float32).numpy():e}, '
            f'Loss: {train_loss.result().numpy():.5f}, '
            f'MSE: {train_mse.result().numpy():e}, '
            f'Time: {time_end - time_start:.2f}')
        
        # Validate the model every test_freq epochs
        if (epoch % test_freq) == 0 or epoch == epochs:
            test_loss.reset_states()
            test_mse.reset_states()
            for test_data, test_labels in test_ds:
                predictions, emotion_input = model(test_data, training=False)
                t_loss = loss_object(test_labels, (predictions, emotion_input)) # Update loss
                test_loss.update_state(t_loss)  # Update test metric
                test_mse.update_state(test_labels, predictions) 
            logger.info(    # Log the test information
                f'----- Test '
                f'Loss: {test_loss.result().numpy():.5f}, '
                f'MSE: {test_mse.result().numpy():e}')

        # Save the model every save_freq epochs
        if (epoch % save_freq) == 0 or epoch == epochs:
            model.save_weights(checkpoint_save_path)
            logger.info(f"----- Save Checkpoint: {checkpoint_save_path}")
    model.summary() # Print the model summary
    model.save(model_save_path) # Save the model


if __name__ == '__main__':
    # Set random seed
    tf.random.set_seed(123)
    dataSet = 'dataSet4_6'

    # Training Parameters
    EPOCHS = 200    # The number of epochs to train the model
    CKPT_EPOCHS = 0 # The epoch to restore the model
    
    test_freq = 10  # Test the model every test_freq epochs
    save_freq = 10  # Save the model every save_freq epochs
    batch_size = 32 # Batch size
    initial_learning_rate = 0.001   # Initial learning rate

    keep_pro = 0.5  # Dropout rate

    # Path
    project_dir = './'
    output_path = './output4_6/'

    checkpoint_save_path = output_path + 'checkpoint/Audio2Face'
    model_save_path = output_path + 'models/Audio2Face'

    # Create output folder
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    data_dir = os.path.join(project_dir, dataSet)
    logger = get_logger(filename=os.path.join(output_path, 'log.txt'))

    # Load data
    x_train = np.load(os.path.join(data_dir, 'train_data.npy'))
    x_val = np.load(os.path.join(data_dir, 'val_data.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_label_var.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_label_var.npy'))

    output_size = y_val.shape[1]

    # Convert to tensor
    x_train = tf.convert_to_tensor(value=x_train, dtype=tf.float32)[:-1]
    y_train = tf.convert_to_tensor(value=y_train, dtype=tf.float32)[:-1]
    x_val = tf.convert_to_tensor(value=x_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(value=y_val, dtype=tf.float32)

    # Create dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    # Setting training schedule
    schedules = tf.keras.optimizers.schedules.ExponentialDecay(
                                                initial_learning_rate,
                                                decay_steps=x_train.shape[0] // batch_size * 10,
                                                decay_rate=0.98,
                                                staircase=True)
    # Setting optimizer
    optimizer = tf.keras.optimizers.Adam(schedules, epsilon=1e-08)
    
    # Setting loss function
    loss_object = losses

    # Build model
    model = Audio2Face(output_size, keep_pro)
    logger.info(f'\n\n--------------------------------------------------------------')

    # Satrt training
    train(EPOCHS, CKPT_EPOCHS, model, checkpoint_save_path, loss_object,
          optimizer, train_ds, test_ds, test_freq, save_freq, logger)
          
    # Train Finished
    logger.info(f'Train Finished')
