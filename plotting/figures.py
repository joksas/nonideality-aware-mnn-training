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


def accuracy_curves():
    spacing = 1
    colors = utils.color_dict()
    # Create two subplots and unpack the output array immediately
    f, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18/2.54, 5/2.54))

    iterators = simulations.iv_nonlinearity.get_iterators()[:3]
    num_epochs = len(iterators[0].info()["history"]["accuracy"])
    epochs = np.arange(1, num_epochs+1, spacing)

    for i, iterator in enumerate(iterators):
        axis = axes[i]

        train_error = 100*(1 - np.array(iterator.info()["history"]["accuracy"][::spacing]))
        validation_error = 100*(1 - np.array(iterator.info()["history"]["val_accuracy"][::spacing]))

        axis.plot(epochs, train_error, color=colors["blue"])
        axis.plot(epochs, validation_error, color=colors["orange"])
        trans = mtransforms.ScaledTranslation(-16/72, 2/72, f.dpi_scale_trans)
        axis.text(0.0, 1.0, chr(i+65), transform=axis.transAxes + trans, fontweight="bold", fontsize=SUBPLOT_LABEL_SIZE)

    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    plt.xlim([0, num_epochs])
    plt.semilogy()
    axes[1].set_xlabel("Epoch (#)", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[0].set_ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.savefig("error-curves.pdf", bbox_inches='tight')
