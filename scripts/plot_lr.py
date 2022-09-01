import json
import os

import matplotlib.pyplot as plt
import numpy as np


def compare_plots(paths, names, window=None, **kwargs):
    data = []
    for path in paths:
        with open(path, mode='r') as f:
            x, y = json.load(f)
            data.append((x, y))

    if window is not None:
        fig, [ax1, ax2] = plt.subplots(1, 2)
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate')
        ax1.set_yscale('log')
        ax1.set_ylabel('Loss')
        ax1.set_title('With Exponential Smoothing', fontsize=10)
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.set_ylabel('Loss')
        ax2.set_title('With Exponential Smoothing + Moving Average', fontsize=10)
        for x, y in data:
            average_y = []
            for ind in range(len(y) - window + 1):
                average_y.append(np.mean(y[ind:ind + window]))

            for ind in range(window - 1):
                average_y.insert(0, np.nan)

            ax1.plot(x, y)
            ax2.plot(x, average_y)

        ax1.grid()
        ax2.grid()

    else:
        fig = plt.figure()
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.yscale('log')
        plt.ylabel('Loss')
        for x, y in data:
            plt.plot(x, y)
        plt.grid()
    names = [name.replace('.json', '').replace('_', ' ') for name in names]
    fig.suptitle(' vs '.join(names), fontsize=14)
    fig.legend(names)
    plt.show()


def plot(json_file, window=None, name=None):
    with open(json_file, mode='r') as f:
        x, y = json.load(f)

    if window is not None:
        fig, [ax1, ax2] = plt.subplots(1, 2)

        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate')
        ax1.set_yscale('log')
        ax1.set_ylabel('Loss')
        ax1.set_title('With Exponential Smoothing', fontsize=10)
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.set_ylabel('Loss')
        ax2.set_title('With Exponential Smoothing + Moving Average', fontsize=10)

        average_y = []
        for ind in range(len(y) - window + 1):
            average_y.append(np.mean(y[ind:ind + window]))

        for ind in range(window - 1):
            average_y.insert(0, np.nan)

        ax1.plot(x, y)
        ax2.plot(x, average_y)

        ax1.grid()
        ax2.grid()

    else:
        fig = plt.figure()
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.yscale('log')
        plt.ylabel('Loss')

        plt.plot(x, y)

        plt.grid()

    fig.suptitle(name.replace('.json', '').replace('_', ' '), fontsize=14)
    plt.show()
    return fig


def main():
    os.chdir('../')
    root_dir = os.getcwd()
    lr_load_dir = os.path.join(root_dir, 'snapshots', 'data', 'LR')
    file_list = []
    with os.scandir(lr_load_dir) as it:
        for entry in it:
            if '.' in entry.name:
                file_list.append((entry.name, entry.path))

    lr_save_dir = os.path.join(root_dir, 'snapshots', 'data', 'LR figs')
    os.makedirs(lr_save_dir, exist_ok=True)
    compare = input("Do you want to compare multiple graphs? [y/n]\n")
    window = input("Do you want to also show the moving average? If so, input window size:\n")
    save = input("Do you want to save the figure? [y/n]\n")
    if not (compare == 'y' and save == 'y'):
        try:
            window = int(window)
        except ValueError:
            window = None

        if compare == 'n':
            file = input("Select a data file from the list:\n    " +
                         '\n    '.join([f"{i + 1}) {name}" for i, (name, _) in enumerate(file_list)]) + '\n')
            try:
                name, path = file_list[int(file) - 1]
                fig = plot(path, window, name=name)
                if save == 'y':
                    fig.savefig(os.path.join(lr_save_dir, name.replace('.json', '.png')))
            except IndexError or ValueError:
                print('Input a valid number from the list')
                raise

        elif compare == 'y':
            file_idxs = input("Input a list of numbers delimited by , for each file to compare from the list:\n    " +
                              '\n    '.join([f"{i + 1}) {name}" for i, (name, _) in enumerate(file_list)]) + '\n')
            try:
                file_idxs = file_idxs.split(",")
                (names, paths) = ([file_list[int(j) - 1][i] for j in file_idxs] for i in range(0, 2))

                compare_plots(paths, names, window)

            except ValueError or IndexError:
                print('Input a valid number from the list')
                raise
        return

    raise ValueError("To save comparison plots, use the gui environment")


if __name__ == '__main__':
    main()
