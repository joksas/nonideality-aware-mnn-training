from training.iterator import Iterator, Training, IVNonlinearity, Inference, StuckAtGMin, D2DLognormal
import numpy as np
import matplotlib.pyplot as plt
import simulations
from . import utils

AXIS_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
TICKS_FONT_SIZE = 8
SUBPLOT_LABEL_SIZE = 12
LINEWIDTH=0.75


def iv_nonlinearity_error_curves():
    spacing = 1
    num_rows = 2
    num_cols = 3
    colors = utils.color_dict()
    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(18/2.54, 9/2.54))

    temp_iterators = simulations.iv_nonlinearity.get_iterators()
    iterators = np.array([[temp_iterators[idx] for idx in row] for row in 
        [
            [0, 1, 2],
            [0, 3, 4],
            ]
            ])
    ### test_histories = np.array([[iterators[i, j].info()["callback_infos"][0]["history"][idx] for j, idx in enumerate(row)]
    ###     for i, row in enumerate([
    ###             [0, 0, 0],
    ###             [1, 0, 0],
    ###             ])
    ###         ])
    # Old setup
    test_histories = np.array([[iterators[i, j].info()["callback_infos"][idx]["history"] for j, idx in enumerate(row)]
        for i, row in enumerate([
                [0, 0, 0],
                [1, 0, 0],
                ])
            ])
    num_epochs = len(iterators[0, 0].info()["history"]["accuracy"])
    epochs = np.arange(1, num_epochs+1, spacing)

    for i in range(num_rows):
        for j in range(num_cols):
            iterator = iterators[i, j]
            test_history = test_histories[i, j]
            test_epochs = test_history["epoch_no"]
            test_accuracy = np.array(test_history["accuracy"])
            test_error = 100*(1 - test_accuracy)
            test_error_median = np.median(test_error, axis=1)
            test_error_p10 = np.quantile(test_error, 0.1, axis=1)
            test_error_p90 = np.quantile(test_error, 0.9, axis=1)
            axis = axes[i, j]

            train_error = 100*(1 - np.array(iterator.info()["history"]["accuracy"][::spacing]))
            validation_error = 100*(1 - np.array(iterator.info()["history"]["val_accuracy"][::spacing]))

            axis.plot(epochs, train_error, color=colors["orange"], linewidth=LINEWIDTH)
            axis.plot(epochs, validation_error, color=colors["sky-blue"], linewidth=LINEWIDTH)
            axis.plot(test_epochs, test_error_median, color=colors["reddish-purple"], linewidth=LINEWIDTH)
            utils.add_subfigure_label(fig, axis, i*num_cols+j, SUBPLOT_LABEL_SIZE)
            plt.setp(axis.get_xticklabels(), fontsize=TICKS_FONT_SIZE)
            plt.setp(axis.get_yticklabels(), fontsize=TICKS_FONT_SIZE)
            axis.set_yscale("log")

            if i+1 == num_rows:
                axes[i, j].set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)

            if j == 0:
                axes[i, j].set_ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.xlim([0, num_epochs])
    plt.figlegend(["Training", "Validation", "Test (median, nonideal)"], ncol=3,
            bbox_to_anchor=(0, 0, 0.85, 1.05), frameon=False)

    plt.savefig("plotting/error-curves.pdf", bbox_inches="tight")

def cnn_results():
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(18/2.54, 5.0/2.54))
    fig.tight_layout()
    colors = utils.color_dict()
    iterators = simulations.iv_nonlinearity_cnn.get_iterators()

    # Error curves.
    for axis, iterator in zip(axes, iterators):
        train_error = 100*(1 - np.array(iterator.info()["history"]["accuracy"]))
        validation_error = 100*(1 - np.array(iterator.info()["history"]["val_accuracy"]))
        num_epochs = len(train_error)
        epochs = np.arange(1, num_epochs+1)
        axis.plot(epochs, train_error, color=colors["orange"], linewidth=LINEWIDTH)
        axis.plot(epochs, validation_error, color=colors["sky-blue"], linewidth=LINEWIDTH)
        axis.set_xlim([0, num_epochs])
        axis.set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)

    axes[0].set_ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    # Box plots.
    axis = axes[2]
    accuracies = [iterator.acc()[0].flatten() for iterator in iterators]
    errors = [100*(1-accuracy) for accuracy in accuracies]
    for idx, (error, color) in enumerate(zip(errors, [colors["vermilion"], colors["blue"]])):
        bplot = axis.boxplot(error, positions=[idx])
        plt.setp(bplot["fliers"], marker="x", markersize=1, markeredgewidth=0.2)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.2)
        plt.xticks([0, 1], ["Standard", "Nonideality-aware"], fontsize=TICKS_FONT_SIZE)
        axis.set_xlabel("Training", fontsize=AXIS_LABEL_FONT_SIZE)

    # Common properties.
    for idx, axis in enumerate(axes):
        axis.set_yscale("log")
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)
        plt.setp(axis.get_xticklabels(), fontsize=TICKS_FONT_SIZE)
        # TODO: Why doesn't y tick label size change??
        plt.setp(axis.get_yticklabels(), fontsize=TICKS_FONT_SIZE)

    plt.figlegend(["Training", "Validation"], ncol=2, bbox_to_anchor=(0, 0, 0.55, 1.15), frameon=False)

    plt.savefig("plotting/cnn-boxplots.pdf", bbox_inches="tight")
