#!/usr/bin/env python3.7

"""Premise selection model."""

# -- Build-in modules --
from argparse import ArgumentParser
import os

# -- Third-party modules --
import numpy as np
import pickle
from tensorflow.keras.layers import Add, BatchNormalization, Input, Dense, Dropout, Flatten, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# -- File info --
__version__ = '0.2.5'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-09-11'

# Argument parser
parser = ArgumentParser(description='Process arguments')
# - Path to data directory
parser.add_argument('-d',
                    '--data_dir',
                    required=True,
                    help='Path to training and test data.',
                    type=str,
                    default=None)
# - Training parameters
parser.add_argument('-bs',
                    '--batch_size',
                    required=False,
                    help='Batch size.',
                    type=int,
                    default=32)
parser.add_argument('-e',
                    '--epochs',
                    required=False,
                    help='Number of epochs.',
                    type=int,
                    default=10)
parser.add_argument('-val',
                    '--validation',
                    required=False,
                    help='Validation split',
                    type=float,
                    default=0)
# - Model architecture parameters
parser.add_argument('-lc',
                    '--layers_config',
                    required=False,
                    help='Number of units in each layer (separate with commas).',
                    type=str,
                    default=None)
parser.add_argument('-res',
                    '--res',
                    required=False,
                    help='Residual connection?',
                    type=str,
                    default='False')
parser.add_argument('-reg',
                    '--reg',
                    required=False,
                    help='Regularize?',
                    type=str,
                    default='False')


def build_premise_selection_model(input_shape: tuple, layers_config: list = None, res: bool = False, reg: bool = False):
    """
    Premise selection model.

    Parameters
    ----------
    input_shape : tuple
        shape of the input premises; the length of the conjecture vector plus the length of the axiom vector,
    layers_config : list
        configuration of the dense layers; the i-th element of layers_config correspond to the number of dense units
        of the i-th dense layer,
    res : bool
        True if each dense layer is to be followed by a residual connection with another dense layer, with the same
        number of parameters,
    reg : bool
        True if there is to be a regularization (batch normalization, bias l1_l2 regularizer, dropout).

    Returns
    -------
    model
        Keras model; premise selection network.
    """
    if layers_config is None:
        layers_config = []

    if reg:
        regularizer = reg = l1_l2(l1=0.01, l2=0.01)
    else:
        regularizer = None

    model_input = Input(shape=input_shape, name='input_layer')
    x = Flatten(name='flatten')(model_input)

    for n in range(len(layers_config)):
        x = Dense(layers_config[n], bias_regularizer=regularizer, name='dense_' + str(n))(x)
        if reg:
            x = BatchNormalization(name='batch_normalization_' + str(n))(x)
            x = ReLU(name='re_lu_' + str(n))(x)
            x = Dropout(0.5, name='dropout_' + str(n))(x)
        else:
            x = ReLU(name='re_lu_' + str(n))(x)
        if res:
            y = Dense(layers_config[n], bias_regularizer=regularizer, name='res_dense_' + str(n))(x)
            x = Add(name='add_' + str(n))([x, y])
            if reg:
                x = BatchNormalization(name='res_batch_normalization_' + str(n))(x)
                x = ReLU(name='res_re_lu_' + str(n))(x)
                x = Dropout(0.5, name='res_dropout_' + str(n))(x)
            else:
                x = ReLU(name='res_re_lu_' + str(n))(x)

    model_output = Dense(1, activation='sigmoid', name='dense_' + str(len(layers_config)))(x)

    model = Model(model_input, model_output, name='premise_selection_model')

    return model


def main():
    # Arguments
    args = vars(parser.parse_args())
    # - Path to data directory
    data_dir = args['data_dir']
    # - Training parameters
    batch_size = args['batch_size']
    epochs = args['epochs']
    val = args['validation'] if args['validation'] != 0 else None
    # - Model architecture parameters
    layers_config = [] if (args['layers_config'] is None) else [int(unit) for unit in args['layers_config'].split(',')]
    res = True if (args['res'].lower() in ['t', 'true', '1']) else False
    reg = True if (args['reg'].lower() in ['t', 'true', '1']) else False

    # Checks
    if not os.path.isdir(data_dir):
        exit('Path to data directory is not a valid path!')

    # Data
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Assertions
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(x_test)

    # Regularize the data
    mean = x_train.mean(axis=0)
    x_train -= mean
    x_test -= mean

    std = x_train.std(axis=0)
    x_train /= std
    x_test /= std

    # Model
    model = build_premise_selection_model(input_shape=x_train.shape[1:], layers_config=layers_config, res=res, reg=reg)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    model.summary()

    # Train model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=val)

    # Save history
    with open('histories/batch_size={}_epochs={}_val={}_'.format(batch_size, epochs, val)
              + 'layers_config={}_res={}_reg={}.pickle'.format(layers_config, res, reg), 'wb') as dictionary:
        pickle.dump(history.history, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    # Save trained model
    model.save('models/rnn_epochs={}_batch_size={}_val={}_'.format(batch_size, epochs, val)
               + 'layers_config={}_res={}_reg={}.h5'.format(layers_config, res, reg))

    # Test model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss {}, test accuracy: {}%.'.format(round(test_loss, 4), round(100 * test_acc, 2)))


if __name__ == '__main__':
    main()
