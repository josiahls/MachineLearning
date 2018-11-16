import itertools
from logging import *

import sys
import numpy as np
import matplotlib.pyplot as plt

basicConfig(stream=sys.stderr, level=DEBUG)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_precision_recall_f1(metrics, metric_labels,
                             title='Metrics for: ',
                             cmap=plt.cm.Blues):
    metrics = np.array(metrics[:-1]).reshape(-1, 1).T

    plt.imshow(metrics, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    tick_marks = np.arange(len(metric_labels))
    plt.xticks(tick_marks, metric_labels, rotation=45)
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False)

    fmt = '.2f'
    thresh = np.average(metrics)
    for i, j in itertools.product(range(metrics.shape[0]), range(metrics.shape[1])):
        plt.text(j, i, format(metrics[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if metrics[i, j] > thresh else "black")

    plt.tight_layout()


def plot_k_folds(k_folds: int, best_param_per_iter: np.array, rmse_test_per_iter: np.array,
                 rmse_train_per_iter: np.array, cost_log_per_iter: np.array, title='',
                 cmap=plt.cm.Blues):
    metrics_list = []
    for i in range(len(best_param_per_iter)):
        metrics_list.append(np.array([i % k_folds, rmse_test_per_iter[i][-1], rmse_train_per_iter[i][-1],
                                      cost_log_per_iter[i][-1], 0]))

    metrics = np.array(metrics_list)
    metric_labels = ['K', 'RMSE for Test', 'RMSE for Train', 'Cost', 'Params']

    title = get_formatted_params(best_param_per_iter[0], 'Params are shortened to \n')

    height = int(2 * len(best_param_per_iter) * (len(title) * .0001 + 1))
    # print(height)
    debug(f'height for k fold figure is {height}')
    plt.figure(figsize=(8, height))
    plt.imshow(metrics, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    tick_marks = np.arange(len(metric_labels))
    plt.xticks(tick_marks, metric_labels, rotation=45)
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False)

    fmt = '.2f'
    thresh = np.average(metrics)
    for i, j in itertools.product(range(metrics.shape[0]), range(metrics.shape[1])):
        if j != metrics.shape[1] - 1:
            plt.text(j, i, format(metrics[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if metrics[i, j] > thresh else "black")
        else:
            s = ['' + str(get_shortened_key(param)) + ':' + str(best_param_per_iter[i][param]) for param in
                 best_param_per_iter[i]]
            plt.text(j, i, ',\n'.join(s),
                     horizontalalignment="center",
                     verticalalignment='center',
                     color="black")

    plt.tight_layout()


def get_formatted_params(parms: dict, prefix='', include_values=False):
    s = prefix
    for key in parms:
        if include_values:
            s += get_shortened_key(key) + ':' + str(parms[key]) + '\n'
        else:
            s += key + ' : ' + get_shortened_key(key) + '\n'
    return s


def get_shortened_key(k: str):
    """ Gets the first letter of each word, and returns the acronym """
    return ''.join([m[0].upper() for m in k.split('_')])
