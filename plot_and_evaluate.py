#!/usr/bin/env python3.7

"""Plot training and validation losses and accuracy."""

# -- Built-in modules --
import ast
import os
from argparse import ArgumentParser

# -- Third-party modules --
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

# -- Proprietary modules --
from rnn_network_model import build_rnn_model

# -- File info --
__version__ = '0.1.4'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-09-18'

# Argument parser
parser = ArgumentParser(description='Process arguments')
parser.add_argument('-hd',
                    '--hist_dir',
                    required=False,
                    help='Path to the histories directory.',
                    type=str,
                    default=None)
parser.add_argument('-md',
                    '--model_dir',
                    required=False,
                    help='Path to the models directory.',
                    type=str,
                    default=None)
parser.add_argument('-d',
                    '--data_dir',
                    required=False,
                    help='Path to the data directory.',
                    type=str,
                    default=None)
parser.add_argument('-pd',
                    '--plot_dir',
                    required=False,
                    help='Path to the plots directory.',
                    type=str,
                    default=None)


def get_title(file_name):
    if 'embedding' in file_name:
        title = 'embedding'
    else:
        print(file_name)
        title = [lc for lc in file_name.split('_') if 'lc' in lc.lower()][0].split('=')[-1][1:-1].replace(' ', '')

        if 'res=True' in file_name:
            res = True
        else:
            res = False

        if res:
            title = title.split(',')
            title = u'\u00D7'.join([number + '(+' + number + ')' for number in title])
        else:
            title = title.replace(',', u'\u00D7')

        if title == '':
            title = 'logistic regression'
        else:
            title += ' hidden units'

        if 'rnn=True' in file_name:
            title += ' (RNN)'

    title += ' model'

    return title


def plot_graphs(path_to_history, path_to_plots):
    with open(path_to_history, 'rb') as dictionary:
        history = pickle.load(dictionary)

    loss = history['loss']
    acc = history['accuracy']

    try:
        val_loss = history['val_loss']
        val_acc = history['val_accuracy']
    except KeyError:
        val_loss = None
        val_acc = None

    epochs = range(1, len(loss) + 1)

    file_name = ''.join(os.path.basename(path_to_history).split('.')[:-1])
    plot_title = get_title(file_name)

    # Loss
    plt.figure(figsize=(16, 12))
    plt.plot(epochs, loss, 'ro', label='Training loss')
    title_loss = 'Training '
    if val_loss is not None:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        title_loss += 'and validation '
    title_loss += 'loss for the ' + plot_title
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title(title_loss, fontsize=20)
    plt.grid()
    plt.legend(fontsize=16)
    plt.savefig(fname=os.path.join(path_to_plots, file_name) + '_loss.png', bbox_inches='tight')
    plt.close()

    # Accuracy
    plt.figure(figsize=(16, 12))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    title_acc = 'Training '
    if val_acc is not None:
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        title_acc += 'and validation '
    title_acc += 'accuracy for the ' + plot_title
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title(title_acc, fontsize=20)
    plt.grid()
    plt.legend(fontsize=16)
    plt.savefig(fname=os.path.join(path_to_plots, file_name) + '_accuracy.png', bbox_inches='tight')
    plt.close()


def test_models(path_to_model, x_test, y_test):
    try:
        model = load_model(path_to_model)
    except ValueError:
        model_name = str(os.path.basename(path_to_model))[:-3]
        model_name = model_name.split('_')
        config = [chunk.split('=')[-1] for chunk in model_name if chunk.lower().startswith('config')][0]
        config = ast.literal_eval(config)
        bi = [chunk.split('=')[-1] for chunk in model_name if chunk.lower().startswith('bi')][0]
        bi = True if bi == 'True' else False
        rec = [chunk.split('=')[-1] for chunk in model_name if chunk.lower().startswith('rec')][0]
        reg = [chunk.split('=')[-1] for chunk in model_name if chunk.lower().startswith('reg')][0]
        reg = True if reg == 'True' else False
        res = [chunk.split('=')[-1] for chunk in model_name if chunk.lower().startswith('res')][0]
        res = True if res == 'True' else False
        model = build_rnn_model(input_shape=(2, 64, 256), bidirectional=bi, layers_config=config,
                                rec=rec, res=res, reg=reg)
        model.load_weights(path_to_model, by_name=True)
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    model.summary()

    title = get_title(path_to_model)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)[:, 0]

    true_positive = np.dot((y_pred >= .5).astype('float32'), (y_test == True).astype('float32')) / len(y_test)
    true_negative = np.dot((y_pred < .5).astype('float32'), (y_test == False).astype('float32')) / len(y_test)
    false_positive = np.dot((y_pred >= .5).astype('float32'), (y_test == False).astype('float32')) / len(y_test)
    false_negative = np.dot((y_pred < .5).astype('float32'), (y_test == True).astype('float32')) / len(y_test)

    print('Test loss {}, test accuracy: {}% for {}.'.format(round(test_loss, 4), round(100 * test_acc, 2), title))
    print('Confusion matrix for {}: TP = {}%, TN = {}%, FP = {}%, FN = {}%.'.format(title,
                                                                                    round(100 * true_positive, 2),
                                                                                    round(100 * true_negative, 2),
                                                                                    round(100 * false_positive, 2),
                                                                                    round(100 * false_negative, 2)))


def main():
    # Arguments
    args = vars(parser.parse_args())
    hist_dir = args['hist_dir']
    model_dir = args['model_dir']
    data_dir = args['data_dir']
    plot_dir = args['plot_dir']
    if plot_dir is None:
        try:
            os.mkdir('plots')
        except OSError:
            pass
        plot_dir = os.path.join(os.getcwd(), 'plots')

    if hist_dir is not None:
        for file in os.listdir(hist_dir):
            plot_graphs(path_to_history=os.path.join(hist_dir, file), path_to_plots=plot_dir)

    if not (data_dir is None or model_dir is None):
        mean = np.load(os.path.join(data_dir, 'x_train.npy'), mmap_mode='r').mean(axis=0)
        std = np.load(os.path.join(data_dir, 'x_train.npy'), mmap_mode='r').std(axis=0)
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'), mmap_mode='r')
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'), mmap_mode='r')
        x_test = x_test - mean
        x_test /= std

        for file in os.listdir(model_dir):
            if file.lower().startswith('rnn'):
                x_test = np.load(os.path.join(data_dir, 'x_test_rnn.npy'), mmap_mode='r')
                y_test = np.load(os.path.join(data_dir, 'y_test_rnn.npy'), mmap_mode='r')

            test_models(path_to_model=os.path.join(model_dir, file), x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    main()
