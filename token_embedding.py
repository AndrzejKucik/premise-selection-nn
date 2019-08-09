#!/usr/bin/env python3.7

"""Token embedding for functional signatures, using probabilistic distribution of features."""

# Build-in modules
import argparse
import os

# Third-party modules
import numpy as np
import pickle
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# -- File info --
__version__ = '0.2.1'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-08-09'


def embed_functions(path_to_output, path_to_models, path_to_histories=None, autoencoder=False,
                    dimension=256, epochs=150, batch_size=2048):
    """Function projecting functional signatures of premises into a lower dimensional space.
       Arguments:
       path_to_output - path to premise count functional signatures
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
    output_tensor = np.load(path_to_output, mmap_mode='r')

    # Number of unique functions
    num_fun = output_tensor.shape[-1]

    # Define model parameters
    if autoencoder:
        output_tensor = output_tensor/ np.max(output_tensor, axis=-1)
        input_tensor = output_tensor
        activation = 'sigmoid'
        loss = 'mse'
    else:
        input_tensor = np.identity(num_fun)
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
    history = model.fit(input_tensor, output_tensor, epochs=epochs, batch_size=batch_size, shuffle=True)

    # Save history
    if path_to_histories is not None:
        with open(os.path.join(path_to_histories, 'embedding_model_history.pickle'), 'wb') as dictionary:
            pickle.dump(history.history, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    # Save trained model
    model.save(os.path.join(path_to_models, 'embedding_model.h5'))


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
    if not os.path.isfile(functions_path):
        exit('Path to functions is not a file!')
    if not functions_path[-4:] == '.npy':
        exit('Path to functions is not a numpy array file!')
    if type(dim) is not int:
        exit('Dimension must be an integer.')
    if dim <= 0:
        exit('Dimension must be a positive integer.')

    cwd = os.getcwd()
    try:
        os.mkdir('models')
    except OSError:
        pass

    try:
        os.mkdir('histories')
    except OSError:
        pass

    embed_functions(path_to_output=functions_path, path_to_models=os.path.join(cwd, 'models'),
                    path_to_histories=os.path.join(cwd, 'histories'),
                    dimension=dim, epochs=epochs, batch_size=batch_size)
