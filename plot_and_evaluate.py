#!/usr/bin/env python3.7

"""Plot training and validation losses and accuracy."""

# Built-in modules
import os
from argparse import ArgumentParser

# Third-party modules
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.models import load_model

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
parser.add_argument('-hd',
                    '--hist_dir',
                    required=False,
                    help='Path to the histories directory.',
                    type=str,
                    default=None)

parser.add_argument('-m',
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


def get_title(path):
    if 'embedding' in path:
        title = 'the embedding'
    else:
        title = path.split('layers_config=')[-1]
        if 'res=True' in title:
            res = True
        else:
            res = False

        title = title.split('_')[0]

        if res:
            title = title.split(',')
            title = u"\u00D7".join([number + '(+' + number + ')' for number in title])
        else:
            title = title.replace(',', u"\u00D7")

    if 'rnn' in path:
        title += ' (RNN)'

    return title


def plot_graphs(path_to_history):
    try:
        os.mkdir('plots')
    except OSError:
        pass

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

        title = get_title(path_to_history)

        # Loss
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss, 'ro', label='Training loss')
        title_loss = 'Training '
        if val_loss is not None:
            plt.plot(epochs, val_loss, 'r', label='Validation loss')
            title_loss += 'and validation '
        title_loss += 'loss for ' + title + ' model.'
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title_loss)
        plt.grid()
        plt.legend()
        plt.savefig(fname='plots/'+title.lower().replace(' ', '_')+'.png')

        # Accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        title_acc = 'Training '
        if val_acc is not None:
            plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
            title_acc += 'and validation '
        title_acc += 'accuracy for ' + title + ' model.'
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(title_acc)
        plt.grid()
        plt.legend()
        plt.savefig(fname='plots/'+title.lower().replace(' ', '_')+'.png')


def test_models(path_to_model, x_test, y_test):
    model = load_model(path_to_model)

    title = get_title(path_to_model)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss {}, test accuracy: {}% for {} model.'.format(round(test_loss, 4), round(100 * test_acc, 2), title))


def main():
    # Arguments
    args = vars(parser.parse_args())
    hist_dir = args['hist_dir']
    model_dir = args['model_dir']
    data_dir = args['data_dir']

    if hist_dir is not None:
        for file in os.listdir(hist_dir):
            plot_graphs(os.path.join(hist_dir, file))

    if not (data_dir is None or model_dir is None):
        mean = np.load(os.path.join(data_dir, 'x_train.npy'), mmap_mode='r').mean(axis=0)
        std = np.load(os.path.join(data_dir, 'x_train.npy'), mmap_mode='r').std(axis=0)
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'), mmap_mode='r')
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'), mmap_mode='r')
        x_test = x_test - mean
        x_test /= std

        for file in os.listdir(model_dir):
            try:
                test_models(path_to_model=os.path.join(model_dir, file), x_test=x_test, y_test=y_test)
            except ValueError:
                pass


if __name__ == '__main__':
    main()
