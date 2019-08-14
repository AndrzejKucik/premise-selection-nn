#!/usr/bin/env python3.7

"""Premise selection model."""

# Build-in modules
from argparse import ArgumentParser
import os

# Third-party modules
import numpy as np
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Add, BatchNormalization, Input, Dense, Dropout, Flatten, GaussianNoise, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# -- File info --
__version__ = '0.2.1'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-08-09'

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
                    type=str)
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


def build_premise_selection_model(input_shape, layers_config=None, res=False):
    if layers_config is None:
        layers_config = []

    model_input = Input(shape=input_shape)
    x = Flatten()(model_input)

    for n in range(len(layers_config)):
        x = Dense(layers_config[n], name='dense_' + str(n))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.5)(x)
        if res:
            y = Dense(layers_config[n], name='res_dense_' + str(n))(x)
            x = Add(name='add_' + str(n))([x, y])
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
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

    # Checks
    if not os.path.isdir(data_dir):
        exit('Path to data directory is not a valid path!')

    # Conversions
    config = [int(config) for config in layers_config.split(',')]
    if res.lower() in ['t', 'true', '1']:
        res = True
    else:
        res = False

    # Data
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Assertions
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(x_test)

    # Regularize the data. (OPTIONAL)
    mean = x_train.mean(axis=0)
    x_train -= mean
    x_test -= mean

    std = x_train.std(axis=0)
    x_train /= std
    x_test /= std

    # Model
    model = build_premise_selection_model(input_shape=x_train.shape[1:], layers_config=config, res=res)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    model.summary()

    # Train model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1)

    # Save trained model
    model.save(
        'models/epochs={}_batch_size={}_layers_config={}_res={}.h5'.format(epochs, batch_size, layers_config, res))

    # Save history
    with open('histories/epochs={}_batch_size={}_layers_config={}_res={}.pickle'.format(epochs, batch_size,
                                                                                        layers_config, res), 'wb') \
            as dictionary:
        pickle.dump(history.history, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    # Test model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss {}, test accuracy: {}%.'.format(round(test_loss, 4), round(100 * test_acc, 2)))


if __name__ == '__main__':
    main()
