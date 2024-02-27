import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np


def error_plots(errors_mean, errors_output, labels=None, title_mean='Mean error', title_output='Error per output'):
    errors_output = np.asarray(errors_output)
    if not labels:
        labels = np.arange(1, len(errors_output[0])+1)
    fig_mean = plt.figure(figsize=(10, 5))
    plt.plot(errors_mean)
    plt.title(title_mean)
    plt.show()

    # Achsen Aufteilung pro Output Neuron
    color_map = plt.get_cmap('gist_rainbow')
    colors_normalized = colors.Normalize(vmin=0, vmax=len(labels) - 1)
    scalar_map = mplcm.ScalarMappable(norm=colors_normalized, cmap=color_map)
    fig_output = plt.figure(figsize=(10, 5))
    ax = fig_output.add_subplot(111)
    ax.set_prop_cycle(color=[scalar_map.to_rgba(i) for i in range(len(labels))])
    for i in range(len(labels)):
        ax.plot(errors_output[:, i], label=labels[i])
    plt.legend(labels)
    plt.title(title_output)
    plt.show()
    return fig_mean, fig_output


def f1_plots(f1_mean, f1_output, labels=None, title_mean='Mean F1 score', title_output='F1 score per output',
             title_bar='F1 score per output'):
    f1_output = np.asarray(f1_output)
    if not labels:
        labels = np.arange(1, len(f1_output[0]) + 1)

    fig_mean = plt.figure(figsize=(10, 5))
    plt.plot(f1_mean)
    plt.title(title_mean)
    plt.show()

    # Achsen Aufteilung pro Output Neuron
    color_map = plt.get_cmap('gist_rainbow')
    colors_normalized = colors.Normalize(vmin=0, vmax=len(labels) - 1)
    scalar_map = mplcm.ScalarMappable(norm=colors_normalized, cmap=color_map)
    fig_output = plt.figure(figsize=(10, 5))
    ax = fig_output.add_subplot(111)
    ax.set_prop_cycle(color=[scalar_map.to_rgba(i) for i in range(len(labels))])
    for i in range(len(labels)):
        ax.plot(f1_output[:, i], label=labels[i])
    plt.legend(labels)
    plt.title(title_output)
    plt.show()

    # Balkendiagramm pro Output Neuron
    fig_bar = plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(f1_output[-1])), f1_output[-1], align='center', alpha=0.5, color=[0.9, 0.7, 1])
    plt.xticks(np.arange(len(f1_output[-1])), labels)
    plt.title(title_bar)
    plt.show()
    return fig_mean, fig_output, fig_bar
