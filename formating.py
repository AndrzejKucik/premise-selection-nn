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
from utils import (calculate_context_distribution, convert_to_count_signatures, convert_to_integers, embed_integers,
                   extract_premises, form_train_sets, form_train_sets_rnn)

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

    split = 10

    for file_name in ['x_{}.npy'.format(n) for n in range(split)] + ['y_{}.npy'.format(n) for n in range(split)]:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Forming training and test sets...')
            form_train_sets(path_to_data=os.path.join(cwd, 'data'), weight=weight, split=split)

    test_indices = sample(range(split), split // 10)
    print('Test chunk indices:', test_indices)

    x_train = np.concatenate([np.load(os.path.join(cwd, 'data', 'x_{}.npy'.format(n)))
                              for n in range(split) if n not in test_indices])
    y_train = np.concatenate([np.load(os.path.join(cwd, 'data', 'y_{}.npy'.format(n)))
                              for n in range(split) if n not in test_indices])

    np.save(os.path.join(cwd, 'data', 'x_train.npy'), x_train)
    np.save(os.path.join(cwd, 'data', 'y_train.npy'), y_train)

    x_test = np.concatenate([np.load(os.path.join(cwd, 'data', 'x_{}.npy'.format(n)))
                             for n in range(split) if n in test_indices])
    y_test = np.concatenate([np.load(os.path.join(cwd, 'data', 'y_{}.npy'.format(n)))
                             for n in range(split) if n in test_indices])

    np.save(os.path.join(cwd, 'data', 'x_test.npy'), x_test)
    np.save(os.path.join(cwd, 'data', 'y_test.npy'), y_test)

    print('Number of training pairs:', len(y_train))
    print('Number of test pairs:', len(y_test))

    # RNN
    split_rnn = 100
    print('RNN')
    for file_name in ['conjecture_signatures_int.pickle', 'axiom_signatures_int.pickle']:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Tokenizing functions...')
            convert_to_integers(paths_to_signatures=[os.path.join(cwd, 'data', 'conjecture_signatures.pickle'),
                                                     os.path.join(cwd, 'data', 'axiom_signatures.pickle')])

    for file_name in ['conjecture_signatures_int_embed.pickle', 'axiom_signatures_int_embed.pickle']:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Embedding tokenized functions...')
            embed_integers(paths_to_signatures=[os.path.join(cwd, 'data', 'conjecture_signatures_int.pickle'),
                                                os.path.join(cwd, 'data', 'axiom_signatures_int.pickle')],
                           weight=weight)

    for file_name in ['x_rnn_{}.npy'.format(n) for n in range(split_rnn)] + ['y_rnn_{}.npy'.format(n) for n in
                                                                             range(split_rnn)]:
        if file_name not in os.listdir(os.path.join(cwd, 'data')):
            print('Forming training and test sets...')
            form_train_sets_rnn(path_to_data=os.path.join(cwd, 'data'), weight=weight, split=split_rnn, max_len=64)

    test_indices = sample(range(split_rnn), split_rnn // 10)
    print('Test chunk indices:', test_indices)

    x_train = np.concatenate([np.load(os.path.join(cwd, 'data', 'x_rnn_{}.npy'.format(n)))
                              for n in range(split_rnn) if n not in test_indices])
    y_train = np.concatenate([np.load(os.path.join(cwd, 'data', 'y_rnn_{}.npy'.format(n)))
                              for n in range(split_rnn) if n not in test_indices])

    np.save(os.path.join(cwd, 'data', 'x_rnn_train.npy'), x_train)
    np.save(os.path.join(cwd, 'data', 'y_rnn_train.npy'), y_train)

    x_test = np.concatenate([np.load(os.path.join(cwd, 'data', 'x_rnn_{}.npy'.format(n)))
                             for n in range(split_rnn) if n in test_indices])
    y_test = np.concatenate([np.load(os.path.join(cwd, 'data', 'y_rnn_{}.npy'.format(n)))
                             for n in range(split_rnn) if n in test_indices])

    np.save(os.path.join(cwd, 'data', 'x_rnn_test.npy'), x_test)
    np.save(os.path.join(cwd, 'data', 'y_rnn_test.npy'), y_test)

    print('Number of RNN training pairs:', len(y_train))
    print('Number of RNN test pairs:', len(y_test))


if __name__ == '__main__':
    main()
