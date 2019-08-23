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

from tensorflow.keras.layers import (Add, BatchNormalization, Bidirectional, Concatenate, Dense,
                                     Dropout, GRU, Layer, Input, ReLU, LSTM, Reshape, SimpleRNN)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l1_l2

# -- File info --
__version__ = '0.2.0'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-08-23'

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
parser.add_argument('-bi',
                    '--bidirectional',
                    required=False,
                    help='Bidirectional?',
                    type=str,
                    default='False')
parser.add_argument('-lc',
                    '--layers_config',
                    required=False,
                    help='Number of units in each layer (separate with commas).',
                    type=str,
                    default=None)
parser.add_argument('-rec',
                    '--recurrent',
                    required=False,
                    help="Type of recurrent connection ('simple', 'LSTM', or 'GRU')",
                    type=str,
                    default='simple')
parser.add_argument('-res',
                    '--res',
                    required=False,
                    help='Residual connection?',
                    type=str,
                    default='False')


# Split layer
class Split(Layer):
    """Layer splitting the tensor input into a specified number of chunks  along a given axis."""

    def __init__(self, axis=1, num_splits=1, **kwargs):
        """Arguments:
           axis       - integer, axis along which to perform the split,
           num_splits - number of split chunks."""

        self.axis = axis
        self.num_splits = num_splits
        super(Split, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.split(inputs, self.num_splits, axis=self.axis)


def build_rnn_model(input_shape, rec='simple', bidirectional=False, layers_config=None, res=False):
    """Function building a premise selection model.
       Arguments:
       input_shape   - triple of integers: (2, seq_len, seq_dim), where seq_len is the length of the input sequences and
                       seq_dim is the dimension of each of the element of the sequence.
       rec           - type of the recurrent layer; either `lstm`, `gru` or `lstm`,
       bidirectional - Boolean, specifying whether to use a bidirectional wrapper for the recurrent layers,
       layers_config - list of integers, number of units for the consecutive dense layers,
       res           - Boolean, specifying whether to use a residual connection.

       Returns:
       model         - Keras model object.
    """

    # Regularizer
    reg = l1_l2(l1=0.01, l2=0.01)

    # Recurrent layer
    if rec == 'gru':
        recurrent = GRU
    elif rec == 'lstm':
        recurrent = LSTM
    else:  # rec == 'simple'
        recurrent = SimpleRNN

    # Dense layers' configuration
    if layers_config is None:
        layers_config = []

    # Model
    model_input = Input(shape=input_shape, name='conjecture_axiom_input')

    # - Split and reshape
    c, a = Split(num_splits=input_shape[0], axis=1, name='conjecture_axiom_split')(model_input)
    a = Reshape(input_shape[1:], name='axiom_reshape')(a)
    c = Reshape(input_shape[1:], name='conjecture_reshape')(c)

    # - Bidirectional RNN
    if bidirectional:
        # -- Axiom branch
        a = Bidirectional(recurrent(128, return_sequences=True,
                                    recurrent_regularizer=reg, bias_regularizer=reg,
                                    dropout=0.5, recurrent_dropout=0.5,
                                    name='axiom_' + rec + '_0'),
                          name='axiom_bidirectional_' + rec + '_0')(a)
        a = Bidirectional(recurrent(128,
                                    recurrent_regularizer=reg, bias_regularizer=reg,
                                    dropout=0.5, recurrent_dropout=0.5,
                                    name='axiom_' + rec + '_1'),
                          name='axiom_bidirectional_' + rec + '_1')(a)

        # -- Conjecture branch
        c = Bidirectional(recurrent(128, return_sequences=True,
                                    recurrent_regularizer=reg, bias_regularizer=reg,
                                    dropout=0.5, recurrent_dropout=0.5,
                                    name='conjecture_' + rec + '_0'),
                          name='conjecture_bidirectional_' + rec + '_0')(c)
        c = Bidirectional(recurrent(128,
                                    recurrent_regularizer=reg, bias_regularizer=reg,
                                    dropout=0.5, recurrent_dropout=0.5,
                                    name='conjecture_' + rec + '_1'),
                          name='conjecture_bidirectional_' + rec + '_1')(c)

    # - Unidirectional RNN
    else:
        # -- Axiom branch
        a = recurrent(256, return_sequences=True,
                      recurrent_regularizer=reg, bias_regularizer=reg,
                      dropout=0.5, recurrent_dropout=0.5,
                      name='axiom_' + rec + '_0')(a)
        a = recurrent(256,
                      recurrent_regularizer=reg, bias_regularizer=reg,
                      dropout=0.5, recurrent_dropout=0.5,
                      name='axiom_' + rec + '_1')(a)

        # -- Conjecture branch
        c = recurrent(256, return_sequences=True,
                      recurrent_regularizer=reg, bias_regularizer=reg,
                      dropout=0.5, recurrent_dropout=0.5,
                      name='conjecture_' + rec + '_0')(c)
        c = recurrent(256,
                      recurrent_regularizer=reg, bias_regularizer=reg,
                      dropout=0.5, recurrent_dropout=0.5,
                      name='conjecture_' + rec + '_1')(c)

    x = Concatenate(name='conjecture_axiom_concatenate')([c, a])

    # - Dense layers
    for n in range(len(layers_config)):
        x = Dense(layers_config[n], bias_regularizer=reg, name='dense_' + str(n))(x)
        x = BatchNormalization(name='batch_normalization_' + str(n))(x)
        x = ReLU(name='re_lu_' + str(n))(x)
        x = Dropout(0.5, name='dropout_' + str(n))(x)

        # -- Residual layers
        if res:
            y = Dense(layers_config[n], name='res_dense_' + str(n), bias_regularizer=reg)(x)
            x = Add(name='add_' + str(n))([x, y])
            x = BatchNormalization(name='res_batch_normalization_' + str(n))(x)
            x = ReLU()(name='res_re_lu_' + str(n))(x)
            x = Dropout(0.5, name='res_dropout_' + str(n))(x)

    # - Model output; usefulness of an axiom, given the conjecture
    model_output = Dense(1, activation='sigmoid', name='conjecture_usefulness')(x)

    model = Model(model_input, model_output, name='rnn_premise_selection_model')

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
    bi = True if (args['bidirectional'].lower() in ['t', 'true', '1']) else False
    layers_config = [] if (args['layers_config'] is None) else [int(unit) for unit in args['layers_config'].split(',')]
    rec = args['recurrent'].lower()
    res = True if (args['res'].lower() in ['t', 'true', '1']) else False

    # Data chunks
    chunks = list(range(len([file for file in os.listdir(data_dir) if 'x_train_rnn' in file])))

    # Model
    model = build_rnn_model(input_shape=(2, 64, 256), bidirectional=bi, layers_config=layers_config, rec=rec, res=res)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    # Training history
    main_history = {'loss': [], 'accuracy': []}
    if val is not None:
        main_history['val_loss'] = []
        main_history['val_accuracy'] = []

    # Training
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        # Shuffle the chunks before training
        shuffle(chunks)

        # Count the total number of training batches
        total_num_batches = 0

        # Placeholders for loss, accuracy (optionally also: validation loss and validation accuracy)
        loss, accuracy, val_loss, val_accuracy = 0, 0, 0, 0

        # Loop through training data chunks
        for chunk in chunks:
            print('Training of chunk {}.'.format(chunk))
            # Load data
            x_train = np.load(os.path.join(data_dir, 'x_train_rnn_{}.npy'.format(chunk)), mmap_mode='r')
            y_train = np.load(os.path.join(data_dir, 'y_train_rnn_{}.npy'.format(chunk)), mmap_mode='r')

            # Count and update the number of batches
            num_batches = len(y_train) // batch_size
            total_num_batches += num_batches

            # Train the model
            history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, shuffle=True, validation_split=val)
            history_dict = history.history

            # Loss and accuracy are multiplied by the number of batches for weighted average
            loss += history_dict['loss'][0] * num_batches
            accuracy += history_dict['accuracy'][0] * num_batches
            if val is not None:
                val_loss += history_dict['val_loss'][0] * num_batches
                val_accuracy += history_dict['val_accuracy'][0] * num_batches

        # We store weighted average per each epoch
        main_history['loss'].append(loss / total_num_batches)
        main_history['accuracy'].append(accuracy / total_num_batches)
        if val is not None:
            main_history['val_loss'].append(val_loss / total_num_batches)
            main_history['val_accuracy'].append(val_accuracy / total_num_batches)

        # Save history
        with open('histories/rnn_batch_size={}_epochs={}_val={}'.format(batch_size, epochs, val)
                  + 'bi={}_layers_config={}_rec={}_res={}.pickle'.format(bi, layers_config, rec, res),
                  'wb') as dictionary:
            pickle.dump(main_history, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    # Save trained model
    model.save('models/rnn_epochs={}_batch_size={}_val={}'.format(batch_size, epochs, val)
               + 'bi={}_layers_config={}_rec={}_res={}.h5'.format(bi, layers_config, rec, res))

    # Test model
    x_test = np.load(os.path.join(data_dir, 'x_test_rnn.npy'), mmap_mode='r')
    y_test = np.load(os.path.join(data_dir, 'y_test_rnn.npy'), mmap_mode='r')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss {}, test accuracy: {}%.'.format(round(test_loss, 4), round(100 * test_acc, 2)))


if __name__ == '__main__':
    main()
