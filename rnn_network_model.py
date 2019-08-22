#!/usr/bin/env python3.7

"""Premise selection RNN model."""

# -- Build-in modules --
from argparse import ArgumentParser
from random import shuffle
import os

# -- Third-party modules --
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import (Add, BatchNormalization, Bidirectional, Input, Concatenate, Dense, Dropout,
                                     Flatten, Layer, ReLU, LSTM, Reshape, GaussianNoise)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2, l2, l1

def orthogonal_regularizer(scale):
    def ortho_reg(weight):
        weight = tf.reshape(weight, (-1, weight.shape[-1]))

        weight = tf.matmul(weight, weight, transpose_a=True)

        weight = weight - tf.eye(weight.shape[0])

        return scale * tf.nn.l2_loss(weight)

    return ortho_reg

reg = l1_l2(l1=0.01, l2=0.01)
bias_reg = l1_l2(l1=0.01, l2=0.01)
dense_bias_reg = l1_l2(l1=0.01, l2=0.01)

# -- File info --
__version__ = '0.1.0'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-08-21'

# Argument parser
parser = ArgumentParser(description='Process arguments')

# Argument for recording path (SEF format)
parser.add_argument('-d',
                    '--data_dir',
                    required=True,
                    help='Path to training and test data.',
                    type=str,
                    default=None)
parser.add_argument('-lc',
                    '--layers_config',
                    required=False,
                    help='Number of units in each layer (separate with commas).',
                    type=str,
                    default=None)
parser.add_argument('-r',
                    '--res',
                    required=False,
                    help='Residual connection?',
                    type=str,
                    default='False')
parser.add_argument('-e',
                    '--epochs',
                    required=False,
                    help='Number of epochs.',
                    type=int,
                    default=10)
parser.add_argument('-bs',
                    '--batch_size',
                    required=False,
                    help='Batch size.',
                    type=int,
                    default=32)


class Split(Layer):
    def __init__(self, axis=1, num_splits=1, **kwargs):
        self.axis = axis
        self.num_splits = num_splits
        super(Split, self).__init__(**kwargs)

    def call(self, inputs):
        # Expect the input to be 3D and mask to be 2D, split the input tensor into 2 among the `axis`.
        return tf.split(inputs, self.num_splits, axis=self.axis)


def build_rnn_model(input_shape, layers_config=None, res=False):
    if layers_config is None:
        layers_config = []

    model_input = Input(shape=input_shape)

    x = GaussianNoise(stddev=1)(model_input)

    c, a = Split(num_splits=2, axis=1)(x)

    a = Reshape((64, 256))(a)
    c = Reshape((64, 256))(c)

    a = Bidirectional(LSTM(128, return_sequences=True, recurrent_regularizer=reg, bias_regularizer=bias_reg))(a)
    a = Bidirectional(LSTM(128, recurrent_regularizer=reg, bias_regularizer=bias_reg))(a)

    c = Bidirectional(LSTM(128, return_sequences=True, recurrent_regularizer=reg, bias_regularizer=bias_reg))(c)
    c = Bidirectional(LSTM(128, recurrent_regularizer=reg, bias_regularizer=bias_reg))(c)

    x = Concatenate()([c, a])

    for n in range(len(layers_config)):
        x = Dense(layers_config[n], name='dense_' + str(n), bias_regularizer=dense_bias_reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        if res:
            y = Dense(layers_config[n], name='res_dense_' + str(n), bias_regularizer=dense_bias_reg)(x)
            x = Add(name='add_' + str(n))([x, y])
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Dropout(0.5)(x)

    model_output = Dense(1, activation='sigmoid', name='dense_' + str(len(layers_config)))(x)

    model = Model(model_input, model_output)

    return model


def main():
    # Arguments
    args = vars(parser.parse_args())
    data_dir = args['data_dir']
    layers_config = args['layers_config']
    res = args['res']
    epochs = args['epochs']
    batch_size = args['batch_size']

    # Conversions
    if layers_config is None:
        config = []
    else:
        config = [int(config) for config in layers_config.split(',')]
    if res.lower() in ['t', 'true', '1']:
        res = True
    else:
        res = False

    model = build_rnn_model(input_shape=(2, 64, 256), layers_config=config, res=res)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    main_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        split = list(range(10))
        shuffle(split)
        k = 0
        loss, accuracy, val_loss, val_accuracy = 0, 0, 0, 0
        for n in split:
            print('Chunk', n)
            x_train = np.load(os.path.join(data_dir, 'x_train_rnn_{}.npy'.format(n)), mmap_mode='r')
            y_train = np.load(os.path.join(data_dir, 'y_train_rnn_{}.npy'.format(n)), mmap_mode='r')
            k += len(y_train) // batch_size

            history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, shuffle=True, validation_split=0.1)

            history_dict = history.history
            loss += history_dict['loss'][0] * (len(y_train) // batch_size)
            accuracy += history_dict['accuracy'][0] * (len(y_train) // batch_size)
            val_loss += history_dict['val_loss'][0] * (len(y_train) // batch_size)
            val_accuracy += history_dict['val_accuracy'][0] * (len(y_train) // batch_size)

        main_history['loss'].append(loss / k)
        main_history['accuracy'].append(accuracy / k)
        main_history['val_loss'].append(val_loss / k)
        main_history['val_accuracy'].append(val_accuracy / k)

    # Save trained model
    model.save(
        'models/rnn_epochs={}_batch_size={}_layers_config={}_res={}.h5'.format(epochs, batch_size, layers_config, res))

    # Save history
    with open('histories/rnn_epochs={}_batch_size={}_layers_config={}_res={}.pickle'.format(epochs, batch_size,
                                                                                            layers_config, res), 'wb') \
            as dictionary:
        pickle.dump(main_history, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    # Test model
    x_test = np.load(os.path.join(data_dir, 'x_test_rnn.npy'), mmap_mode='r')
    y_test = np.load(os.path.join(data_dir, 'y_test_rnn.npy'), mmap_mode='r')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss {}, test accuracy: {}%.'.format(round(test_loss, 4), round(100 * test_acc, 2)))


if __name__ == '__main__':
    main()
