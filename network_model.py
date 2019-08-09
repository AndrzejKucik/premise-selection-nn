#!/usr/bin/env python3.7

"""Premise selection model."""

# Build-in modules
from argparse import ArgumentParser
from pathlib import Path

# Third-party modules
import numpy as np
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Add, Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

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
parser.add_argument('-r,'
                    '-res',
                    require=False,
                    help='Residual connection?',
                    type=str,
                    default='False')
parser.add_argument('-e',
                    '--epochs',
                    required=False,
                    help='Number of layers.',
                    type=int,
                    default=2)
parser.add_argument('-bs',
                    '--batch_size',
                    required=False,
                    help='Batch size.',
                    type=int,
                    default=64)


def build_premise_selection_model(input_shape, layers_config=None, res=False):

    if layers_config is None:
        layers_config = []

    model_input = Input(shape=input_shape)
    x = Flatten()(model_input)

    for n in range(len(layers_config)):
        x = Dense(layers_config[n], activation='relu', name='dense_' + str(n))(x)
        if res:
            y = Dense(layers_config[n], activation='relu', name='res_dense_' + str(n))(x)
            x = Add(name='add_' + str(n))([x, y])

    model_output = Dense(1, activation='sigmoid', name='dense_' + str(len(layers_config)))(x)

    model = Model(model_input, model_output)

    return model


def main():
    # Arguments
    args = vars(parser)
    data_dir = args['data_dir']
    layers_config = args['layers_config']
    res = args['res']
    epochs = args['epochs']
    batch_size = args['batch_size']

    # Checks
    if not Path(data_dir).is_dir():
        exit('Path to data directory is not a valid path!')

    # Conversions
    config = [int(config) for config in layers_config.split(',')]
    if res.lower() in ['t', 'true', '1']:
        res = True
    else:
        res = False

    # Data
    x_train = np.load(Path.joinpath(data_dir, 'training_data.npy'))
    x_test = np.load(Path.joinpath(data_dir, 'test_data.npy'))
    y_train = np.load(Path.joinpath(data_dir, 'training_labels.npy'))
    y_test = np.load(Path.joinpath(data_dir, 'test_labels.npy'))

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

    # Callbacks
    callbacks_list = [EarlyStopping(monitor='val_acc', patience=25)]

    # Model
    model = build_premise_selection_model(input_shape=x_train.shape[-1], layers_config=config)
    model.summary()

    # Train model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        callbacks=callbacks_list, validation_split=0.1, verbose=1)

    # Save trained model
    model.save('models/{}_{}_{}_{}.h5'.format(epochs, batch_size, layers_config, res))

    # Save history
    with open('histories/{}_{}_{}_{}.pickle'.format(epochs, batch_size, layers_config, res), 'wb') as dictionary:
        pickle.dump(history.history, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    # Test model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss {}, test accuracy: {}..'.format(round(test_loss, 4), round(100 * test_acc, 2))