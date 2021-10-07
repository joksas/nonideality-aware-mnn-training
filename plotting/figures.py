from training.iterator import Iterator, Training, IVNonlinearity, Inference, StuckAtGMin, D2DLognormal
import matplotlib.transforms as mtransforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import simulations
from . import utils

AXIS_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
TICKS_FONT_SIZE = 8
SUBPLOT_LABEL_SIZE = 12
LINEWIDTH=0.75


def accuracy_curves():
    spacing = 1
    num_rows = 2
    num_cols = 3
    colors = utils.color_dict()
    # Create two subplots and unpack the output array immediately
    f, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(18/2.54, 9/2.54))

    temp_iterators = simulations.iv_nonlinearity.get_iterators()
    iterators = np.array([[temp_iterators[idx] for idx in row] for row in 
        [
            [0, 1, 2],
            [0, 3, 4],
            ]
            ])
    test_histories = np.array([[iterators[i, j].info()["callback_infos"][0]["history"][idx] for j, idx in enumerate(row)]
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

            axis.plot(epochs, train_error, color=colors["blue"], linewidth=LINEWIDTH)
            axis.plot(epochs, validation_error, color=colors["bluish-green"], linewidth=LINEWIDTH)
            axis.plot(test_epochs, test_error_median, color=colors["vermilion"], linewidth=LINEWIDTH/2)
            axis.fill_between(test_epochs, test_error_p10, test_error_p90, alpha=0.25,
                    color=colors["vermilion"], linewidth=0)
            # axis.plot(test_epoch, color=colors["vermilion"], linewidth=LINEWIDTH)
            trans = mtransforms.ScaledTranslation(-16/72, 2/72, f.dpi_scale_trans)
            axis.text(0.0, 1.0, chr(i*num_cols+j+65), transform=axis.transAxes + trans, fontweight="bold", fontsize=SUBPLOT_LABEL_SIZE)
            plt.setp(axis.get_xticklabels(), fontsize=TICKS_FONT_SIZE)
            plt.setp(axis.get_yticklabels(), fontsize=TICKS_FONT_SIZE)

            if i+1 == num_rows:
                axes[i, j].set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)

            if j == 0:
                axes[i, j].set_ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.xlim([0, num_epochs])
    plt.semilogy()

    plt.savefig("plotting/error-curves.pdf", bbox_inches='tight')
