#!/usr/bin/env python3.7

"""Token embedding for functional signatures, using probabilistic distribution of features."""

# Build-in modules
import argparse
import os
from pathlib import Path

# Third-party modules
import numpy as np
import pickle
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# Proprietary modules
from utils import make_context_vector

# -- File info --
__version__ = '0.2.0'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-08-05'


def embed_functions(functions_path, context_path, dimension=256,
                    epochs=150, batch_size=2048, save_model=True, save_history=True):
    """Function projecting functional signatures of premises into a lower dimensional space.
       Arguments:
       functions_path - path to functional signatures of premises;
                        numpy array of shape (number of premises, number of unique function symbols),
       context_path   - (optional) path to contextualised functional symbols;
                        numpy array of shape (number of unique function symbols, number of unique function symbols),
                        if not provided then model will train as autoencoder.
       autoencoder    - if True, then autoencoder embedding will be the method of projection,
       dimension      - dimension of projection,
       epochs         - number of training epochs for the embedding model,
       batch_size     - batch size for the training of the embedding model,
       save_model     - if True, then model is saved to
                        path_to_current_working_directory/models/embedding_model.h5,
       save_histories - if True then training history is saved to
                        path_to_current_working_directory/histories/embedding_model.pickle,
    """

    # Get functional signatures
    functions = np.load(functions_path, mmap_mode='r')

    # Number of unique functions
    num_fun = functions.shape[1]

    # Get context vectors
    if context_path is None:
        functions_input = functions_output = functions
        activation = 'relu'
        loss = 'mse'
    else:
        if not Path(context_path).is_file():
            context = make_context_vector(path_to_fun=functions_path,
                                          path_to_context=context_path)
        else:
            context = np.load(context_path, mmap_mode='r')

        functions_input = np.identity(num_fun)
        functions_output = context
        activation = 'softmax'
        loss = 'categorical_crossentropy'

    # Embedding model
    model_input = Input(shape=(num_fun,), name='input')
    embedding = Dense(dimension, kernel_initializer=he_uniform(), activation='tanh', name='embedding')(model_input)
    model_output = Dense(num_fun, kernel_initializer=he_uniform(), activation=activation, name='output')(embedding)
    model = Model(model_input, model_output, name='encoder_decoder')
    model.summary()

    # Compile model
    model.compile(optimizer=RMSprop(decay=1e-8), loss=loss, metrics=['accuracy'])

    # Train model
    history = model.fit(functions_input, functions_output, epochs=epochs, batch_size=batch_size, shuffle=True)

    # Save history
    if save_history:
        with open('histories/embedding_model_history.pickle', 'wb') as dictionary:
            pickle.dump(history.history, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    # Save trained model
    if save_model:
        model.save('models/embedding_model.h5')


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Process arguments')

    # Argument for recording path (SEF format)
    parser.add_argument('-f',
                        '--functions',
                        required=True,
                        help='Path to functional signatures (numpy file)',
                        type=str)
    parser.add_argument('-c',
                        '--context',
                        required=True,
                        help='Path to context vector',
                        type=str,
                        default=None)
    parser.add_argument('-dim',
                        '--dimension',
                        required=False,
                        help='Dimension of the embedding.',
                        type=int,
                        default=256)
    parser.add_argument('-e',
                        '--epochs',
                        required=False,
                        help='Number of training epochs.',
                        type=int,
                        default=150)
    parser.add_argument('-bs',
                        '--batch_size',
                        required=False,
                        help='Embedding batch size.',
                        type=int,
                        default=2048)

    # Parse arguments
    args = vars(parser.parse_args())
    functions_path = args['functions']
    context_path = args['context']
    dim = args['dimension']
    epochs = args['epochs']
    batch_size = args['batch_size']

    # Checks
    if not Path(functions_path).is_file():
        exit('Path to functions is not a file!')
    if not functions_path[-4:] == '.npy':
        exit('Path to functions is not a numpy array file!')
    if type(dim) is not int:
        exit('Dimension must be an integer.')
    if dim <= 0:
        exit('Dimension must be a positive integer.')

    embed_functions(functions_path=functions_path, context_path=context_path, dimension=dim,
                    epochs=epochs, batch_size=batch_size, save_model=True, save_history=True)


if __name__ == '__main__':
    main()
