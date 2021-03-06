#!/usr/bin/env python3.7

"""Tools for premise selection NN framework."""

# Built-in modules
import os
from random import sample

# Third-party modules
import numpy as np
from tensorflow.keras.models import load_model, Model

# Proprietary modules
from token_embedding import embed_functions
from utils import (calculate_context_distribution, convert_to_count_signatures, convert_to_integers, embed_count,
                   embed_integers, extract_premises, form_train_sets, get_test_indices)

# -- File info --
__version__ = '0.1.0'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-08-09'


def main():
    cwd = os.getcwd()

    try:
        os.mkdir('data')
    except OSError:
        pass

    for file_name in ['conjecture_signatures.pickle', 'axiom_signatures.pickle',
                      'useful_axioms.pickle', 'useless_axioms.pickle']:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Extracting premises...')
            extract_premises(path_to_premises=os.path.join(cwd, 'nndata'), save_dir=os.path.join(cwd, 'data'))

    for file_name in ['conjecture_signatures_int.pickle', 'axiom_signatures_int.pickle']:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Tokenizing functions...')
            convert_to_integers(paths_to_signatures=[os.path.join(cwd, 'data', 'conjecture_signatures.pickle'),
                                                     os.path.join(cwd, 'data', 'axiom_signatures.pickle')])

    for file_name in ['conjecture_signatures_count.pickle', 'axiom_signatures_count.pickle']:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Counting functions...')
            convert_to_count_signatures(paths_to_signatures=[os.path.join(cwd, 'data', 'conjecture_signatures.pickle'),
                                                             os.path.join(cwd, 'data', 'axiom_signatures.pickle')])

    if 'axiom_signatures_count_context_distribution.npy' not in os.listdir(os.path.join(cwd, 'data')):
        print('Calculating contextual distribution of axiom functional signatures...')
        calculate_context_distribution(path_to_context=os.path.join(cwd, 'data', 'axiom_signatures_count.pickle'))

    try:
        os.mkdir('models')
    except OSError:
        pass

    try:
        os.mkdir('histories')
    except OSError:
        pass

    if 'embedding_model.h5' not in os.listdir(os.path.join(cwd, 'models')):
        embed_functions(path_to_output=os.path.join(cwd, 'data', 'axiom_signatures_count_context_distribution.npy'),
                        path_to_models=os.path.join(cwd, 'models'), path_to_histories=os.path.join(cwd, 'histories'),
                        autoencoder=False, dimension=256, epochs=150, batch_size=2048)

    embedding_model = load_model(os.path.join(cwd, 'models', 'embedding_model.h5'))
    weights = embedding_model.get_layer('embedding').get_weights()
    del embedding_model
    weight = np.tanh(weights[0] + weights[1])

    if 'conjecture_signatures_count_embed.pickle' not in os.listdir(os.path.join(cwd, 'data')):
        print('Embedding conjecture signatures...')
        embed_count(path_to_signatures=os.path.join(cwd, 'data', 'conjecture_signatures_count.pickle'), weight=weight)

    if 'axiom_signatures_count_embed.pickle' not in os.listdir(os.path.join(cwd, 'data')):
        print('Embedding axiom signatures...')
        embed_count(path_to_signatures=os.path.join(cwd, 'data', 'axiom_signatures_count.pickle'), weight=weight)

    if 'conjecture_signatures_int_embed.pickle' not in os.listdir(os.path.join(cwd, 'data')):
        print('Embedding RNN conjecture signatures...')
        embed_integers(path_to_signatures=os.path.join(cwd, 'data', 'conjecture_signatures_int.pickle'), weight=weight)

    if 'axiom_signatures_int_embed.pickle' not in os.listdir(os.path.join(cwd, 'data')):
        print('Embedding RNN axiom signatures...')
        embed_integers(path_to_signatures=os.path.join(cwd, 'data', 'axiom_signatures_int.pickle'), weight=weight)

    for file_name in ['x.npy', 'y.npy']:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Converting the data into numpy arrays...')
            form_train_sets(path_to_data=os.path.join(cwd, 'data'), split=10, rnn=False, embedding_len=weight.shape[1])

    split = 20
    for file_name in ['x_rnn_{}.npy'.format(n) for n in range(split)] +['y_rnn_{}.npy'.format(n) for n in range(split)]:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Converting the data into numpy arrays (RNN)...')
            form_train_sets(path_to_data=os.path.join(cwd, 'data'), split=split, rnn=True,
                            embedding_len=weight.shape[1], max_len=64, concat=False)

    print('Forming training and test sets...')
    x = np.load('data/x.npy', mmap_mode='r')
    y = np.load('data/y.npy', mmap_mode='r')

    test_indices = get_test_indices(os.path.join(cwd, 'data'))
    train_indices = [n for n in range(len(x)) if n not in test_indices]

    np.save('data/x_train.npy', x[train_indices])
    np.save('data/x_test.npy', x[test_indices])
    np.save('data/y_train.npy', y[train_indices])
    np.save('data/y_test.npy', y[test_indices])

    print('Forming training and test sets (RNN)...')
    start = 0
    for k in range(split):
        x = np.load('data/x_rnn_{}.npy'.format(k), mmap_mode='r')
        y = np.load('data/y_rnn_{}.npy'.format(k), mmap_mode='r')
        np.save('data/x_train_rnn_{}.npy'.format(k), x[[n for n in range(len(x)) if n+start in train_indices]])
        np.save('data/y_train_{}.npy'.format(k), y[[n for n in range(len(y)) if n+start in train_indices]])

        if k == 0:
            np.save('data/x_test_rnn.npy', x[[n for n in range(len(x)) if n+start in test_indices]])
            np.save('data/y_test_rnn.npy', y[[n for n in range(len(y)) if n+start in test_indices]])
        else:
            x_test = np.load('data/x_test_rnn.npy', mmap_mode='r')
            x_test = np.concatenate([x_test, x[[n for n in range(len(x)) if n+start in test_indices]]])
            y_test = np.load('data/y_test_rnn.npy', mmap_mode='r')
            y_test = np.concatenate([y_test, y[[n for n in range(len(y)) if n+start in test_indices]]])
            np.save('data/x_test_rnn.npy', x_test)
            np.save('data/y_test_rnn.npy', y_test)

        start += len(x)


if __name__ == '__main__':
    main()
