 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alex Lim
"""

from __future__ import absolute_import, division, print_function
import os
import time
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# Setup paths and directories
PATH_TO_TENSORBOARD_LOG = './logs'
PATH_TO_CHECKPOINTS = './checkpoints'
PATH_TO_SAVED_MODEL = './pretrained_model'

if not os.path.exists(PATH_TO_TENSORBOARD_LOG):
    os.makedirs(PATH_TO_TENSORBOARD_LOG)

if not os.path.exists(PATH_TO_CHECKPOINTS):
    os.makedirs(PATH_TO_CHECKPOINTS)

if not os.path.exists(PATH_TO_SAVED_MODEL):
    os.makedirs(PATH_TO_SAVED_MODEL)

# Load MNIST dataset and normalize pixel values
# X_train: (60000,28,28), X_test: (10000,28,28), y_train: (60000,), y_test: (10000,)
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

# Build the simple dense neural network
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.5, seed=100),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Compile the model using Adam optimizer and sparse_categorical_crossentropy loss
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

# Create the model
model = create_model()

# Print model summary
model.summary()

# Create TensorBoard callback instance
tbCallBack = TensorBoard(log_dir=PATH_TO_TENSORBOARD_LOG, histogram_freq=0,
                            write_graph=True, write_images=True)

# Create Checkpoint callback instance
cpCallBack = ModelCheckpoint(os.path.join(PATH_TO_CHECKPOINTS, 'cp.ckpt'), 
                                save_weights_only=True, verbose=1)

# Train and evaluate the model for 10 epochs
model.fit(X_train, y_train, epochs=3, callbacks=[tbCallBack, cpCallBack])
loss, acc = model.evaluate(X_test, y_test)
print('FINAL MODEL LOSS AND ACCURACY')
print('-----------------------------')
print('Loss: {:.3f}'.format(loss))
print('Accuracy: {:5.2f}%'.format(acc*100))
print('-----------------------------')

# Save the model
print('SAVING MODEL...')
tf.keras.models.save_model(model, os.path.join(PATH_TO_SAVED_MODEL, 'mnist_model.h5'), overwrite=True, include_optimizer=True)
print('MODEL SAVED.)    