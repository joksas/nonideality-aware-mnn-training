import copy

import crossbar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulations
from training import architecture
from training.iterator import (D2DLognormal, Inference, Iterator,
                               IVNonlinearity, StuckAtGMin, Training)

from . import utils

AXIS_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 8
TICKS_FONT_SIZE = 8
SUBPLOT_LABEL_SIZE = 12
LINEWIDTH = 0.75
# Advanced Science
ONE_COLUMN_WIDTH = 8.5 / 2.54
TWO_COLUMNS_WIDTH = 17.8 / 2.54


def iv_nonlinearity_training_curves(metric="error", training_idx=0):
    fig_shape = (2, 3)
    fig, axes = plt.subplots(
        *fig_shape, sharex=True, sharey=True, figsize=(TWO_COLUMNS_WIDTH, TWO_COLUMNS_WIDTH / 2)
    )

    iterators = simulations.iv_nonlinearity.get_iterators()
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx
    # Same training, different inference.
    iterators.insert(3, iterators[0])
    inference_idxs = [0, 0, 0, 1, 0, 0]

    for idx, (iterator, inference_idx) in enumerate(zip(iterators, inference_idxs)):
        i, j = np.unravel_index(idx, fig_shape)
        axis = axes[i, j]
        utils.plot_training_curves(
            fig, axis, iterator, subfigure_idx=idx, metric=metric, inference_idx=inference_idx
        )
        if i + 1 == fig_shape[0]:
            axis.set_xlabel(utils.axis_label("epoch"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)
        if j == 0:
            axis.set_ylabel(utils.axis_label(metric), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    utils.add_legend(fig, ["Training", "Validation", "Test (nonideal)"], ncol=fig_shape[1])

    utils.save_fig(fig, f"iv-nonlinearity-training-{metric}")


def iv_nonlinearity_test(metric="error"):
    fig, axes = plt.subplots(
        figsize=(utils.Config.ONE_COLUMN_WIDTH, 0.8 * utils.Config.ONE_COLUMN_WIDTH)
    )
    fig.tight_layout()
    iterators = simulations.iv_nonlinearity.get_iterators()
    # Same training, different inference.
    iterators.insert(3, iterators[0])
    inference_idxs = [0, 0, 0, 1, 0, 0]

    colors = [utils.color_dict()[key] for key in ["vermilion", "blue", "bluish-green"]]

    boxplots = []

    for idx, (iterator, inference_idx) in enumerate(zip(iterators, inference_idxs)):
        avg_power = iterator.test_metric("avg_power", inference_idx=inference_idx)
        y = iterator.test_metric(metric, inference_idx=inference_idx)
        color = colors[idx % 3]
        boxplot = utils.plot_boxplot(axes, y, color, metric=metric, x=avg_power, is_x_log=True)
        boxplots.append(boxplot)

    utils.add_boxplot_legend(
        axes, boxplots, ["Standard", "Nonideality-aware", "Nonideality-aware (regularised)"]
    )

    plt.xlabel(utils.axis_label("power-consumption"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)
    plt.ylabel(utils.axis_label(metric, prepend="test"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    utils.save_fig(fig, f"iv-nonlinearity-test-{metric}")


def iv_nonlinearity_cnn_results(metric="error", training_idx=0):
    fig_shape = (1, 3)
    fig, axes = plt.subplots(
        *fig_shape, sharey=True, figsize=(TWO_COLUMNS_WIDTH, TWO_COLUMNS_WIDTH / 3)
    )
    fig.tight_layout()

    colors = utils.color_dict()

    iterators = simulations.iv_nonlinearity_cnn.get_iterators()
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx

    axes[0].set_ylabel(utils.axis_label(metric), fontsize=AXIS_LABEL_FONT_SIZE)

    # Error curves.
    for axis, iterator in zip(axes, iterators):
        utils.plot_training_curves(fig, axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    # Box plots.
    axis = axes[2]
    for idx, (iterator, color) in enumerate(zip(iterators, [colors["vermilion"], colors["blue"]])):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Nonideality-aware"])

    axis.set_xlabel("Training", fontsize=AXIS_LABEL_FONT_SIZE)

    # Common properties.
    for idx, axis in enumerate(axes):
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=fig_shape[1],
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, f"iv-nonlinearity-cnn-results-{metric}")


def d2d_conductance_histograms(is_effective=False, include_regularised=False):
    num_rows = 2
    if include_regularised:
        num_rows = 3

    fig, axes = plt.subplots(
        num_rows,
        1,
        sharex=True,
        sharey=True,
        figsize=(ONE_COLUMN_WIDTH, num_rows * 0.75 * ONE_COLUMN_WIDTH),
    )

    iterators = simulations.d2d_asymmetry.get_iterators()
    if not include_regularised:
        iterators = iterators[:2]
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue", "bluish-green"]]

    for idx, (axis, iterator, color) in enumerate(zip(axes, iterators, colors)):
        iterator.is_training = True
        model = architecture.get_model(iterator, custom_weights_path=iterator.weights_path())
        weights = model.layers[1].combined_weights()
        G, _ = crossbar.map.w_params_to_G(weights, iterator.training.G_min, iterator.training.G_max)
        G = 1000 * G
        if is_effective:
            axis.hist(
                G.numpy()[:, ::2].flatten() - G.numpy()[:, 1::2].flatten(),
                bins=100,
                color=color,
            )
        else:
            axis.hist(G.numpy().flatten(), bins=100, color=color)
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)
        axis.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)
        axis.set_ylabel("Count (#)", fontsize=AXIS_LABEL_FONT_SIZE)

    if is_effective:
        label = "Effective conductance (mS)"
        filename = "d2d-G-eff-histograms"
    else:
        label = "Conductance (mS)"
        filename = "d2d-G-histograms"
    if include_regularised:
        filename += "-regularised"
    axes[-1].set_xlabel(label, fontsize=AXIS_LABEL_FONT_SIZE)

    plt.savefig(f"plotting/{filename}.pdf", bbox_inches="tight", transparent=True)


def d2d_conductance_pos_neg_histograms():
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        sharey=True,
        figsize=(ONE_COLUMN_WIDTH, 1.5 * ONE_COLUMN_WIDTH),
    )

    iterators = simulations.d2d_asymmetry.get_iterators()

    for idx, (axis, iterator) in enumerate(zip(axes, iterators)):
        iterator.is_training = True
        model = architecture.get_model(iterator, custom_weights_path=iterator.weights_path())
        weights = model.layers[1].combined_weights()
        G, _ = crossbar.map.w_params_to_G(weights, iterator.training.G_min, iterator.training.G_max)
        G = 1000 * G
        axis.hist(
            G.numpy()[:, ::2].flatten(),
            bins=100,
            color=utils.color_dict()["bluish-green"],
            alpha=0.5,
        )
        axis.hist(
            G.numpy()[:, 1::2].flatten(),
            bins=100,
            color=utils.color_dict()["reddish-purple"],
            alpha=0.5,
        )
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)
        axis.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)
        axis.set_ylabel("Count (#)", fontsize=AXIS_LABEL_FONT_SIZE)

    axes[1].set_xlabel("Conductance (mS)", fontsize=AXIS_LABEL_FONT_SIZE)
    leg = plt.figlegend(
        [r"$G_+$", r"$G_-$"],
        ncol=2,
        bbox_to_anchor=(0, 0, 0.75, 0.95),
        frameon=False,
    )

    plt.savefig("plotting/d2d-G-pos-neg-histograms.pdf", bbox_inches="tight", transparent=True)


def d2d_uniformity_results(metric="error", training_idx=0):
    fig_shape = (1, 3)
    fig, axes = plt.subplots(
        *fig_shape, sharey=True, figsize=(TWO_COLUMNS_WIDTH, TWO_COLUMNS_WIDTH / 3)
    )
    fig.tight_layout()

    colors = utils.color_dict()

    iterators = simulations.d2d_asymmetry.get_iterators()[:2]
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx

    axes[0].set_ylabel(utils.axis_label(metric), fontsize=AXIS_LABEL_FONT_SIZE)

    # Error curves.
    for axis, iterator in zip(axes, iterators):
        utils.plot_training_curves(fig, axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    # Box plots.
    axis = axes[2]
    for idx, (iterator, color) in enumerate(zip(iterators, [colors["vermilion"], colors["blue"]])):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["High", "Low"])
    axis.set_xlabel(utils.axis_label("d2d-uniformity"), fontsize=AXIS_LABEL_FONT_SIZE)

    # Common properties.
    for idx, axis in enumerate(axes):
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=fig_shape[1],
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, f"d2d-uniformity-results-{metric}")


def iv_nonlinearity_and_stuck_results(metric="error", training_idx=0):
    fig_shape = (1, 3)

    fig, axes = plt.subplots(
        *fig_shape,
        sharey=True,
        figsize=(utils.Config.TWO_COLUMNS_WIDTH, utils.Config.TWO_COLUMNS_WIDTH / 3),
    )
    fig.tight_layout()

    iterators = simulations.iv_nonlinearity_and_stuck.get_iterators()
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx

    axes[0].set_ylabel(utils.axis_label(metric), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    # Curves
    for idx, (iterator, axis) in enumerate(zip(iterators, axes)):
        utils.plot_training_curves(fig, axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    # Box plots
    axis = axes[-1]
    boxplots = []
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    for idx, (iterator, color) in enumerate(zip(iterators, colors)):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Nonideality-aware"])
    axis.set_xlabel(utils.axis_label("training"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    for idx, axis in enumerate(axes):
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=fig_shape[1],
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, f"iv-nonlinearity-and-stuck-results-{metric}")


def checkpoint_comparison_boxplots(metric="error", training_idx=0):
    fig_shape = (1, 3)

    fig, axes = plt.subplots(
        *fig_shape,
        sharey=True,
        figsize=(utils.Config.TWO_COLUMNS_WIDTH, utils.Config.TWO_COLUMNS_WIDTH / 3),
    )
    fig.tight_layout()

    iterators = simulations.checkpoint_comparison.get_iterators()
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx

    axes[0].set_ylabel(utils.axis_label(metric), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    # Curves
    for idx, (iterator, axis) in enumerate(zip(iterators, axes)):
        utils.plot_training_curves(fig, axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    # Box plots
    axis = axes[-1]
    boxplots = []
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    for idx, (iterator, color) in enumerate(zip(iterators, colors)):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Memristive"])
    axis.set_xlabel(utils.axis_label("checkpoint"), fontsize=utils.Config.AXIS_LABEL_FONT_SIZE)

    for idx, axis in enumerate(axes):
        utils.add_subfigure_label(fig, axis, idx, SUBPLOT_LABEL_SIZE)

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=fig_shape[1],
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, f"checkpoint-results-{metric}")


def nonideality_agnosticism_heatmap():
    training_labels = {
        "nonreg__64__none_none__ideal": "Ideal",
        "nonreg__64__0.000997_0.00351__IVNL:2.13_0.095": r"Low $I$-$V$ nonlin.",
        "reg__64__0.000997_0.00351__IVNL:2.13_0.095": r"Low $I$-$V$ nonlin. (reg.)",
        "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369": r"High $I$-$V$ nonlin.",
        "reg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369": r"High $I$-$V$ nonlin. (reg.)",
        "nonreg__64__0.000997_0.00351__D2DLN:0.25_0.25": "More uniform D2D var.",
        "nonreg__64__0.000997_0.00351__D2DLN:0.05_0.5": "Less uniform D2D var.",
        "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369+StuckMax:0.05": r"High $I$-$V$ nonlin. + stuck",
    }
    inference_labels = {
        "none_none__ideal": training_labels["nonreg__64__none_none__ideal"],
        "0.000997_0.00351__IVNL:2.13_0.095": training_labels[
            "nonreg__64__0.000997_0.00351__IVNL:2.13_0.095"
        ],
        "7.72e-07_2.73e-06__IVNL:2.99_0.369": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369"
        ],
        "0.000997_0.00351__D2DLN:0.25_0.25": training_labels[
            "nonreg__64__0.000997_0.00351__D2DLN:0.25_0.25"
        ],
        "0.000997_0.00351__D2DLN:0.05_0.5": training_labels[
            "nonreg__64__0.000997_0.00351__D2DLN:0.05_0.5"
        ],
        "7.72e-07_2.73e-06__IVNL:2.99_0.369+StuckMax:0.05": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369+StuckMax:0.05"
        ],
    }
    df = pd.DataFrame(
        index=[training_labels[key] for key in training_labels],
        columns=[inference_labels[key] for key in inference_labels],
    )
    df = df.astype(float)
    iterators = simulations.nonideality_agnosticism.get_iterators()
    for iterator in iterators:
        training_label = training_labels[iterator.training.label()]
        errors = [np.median(error) for error in iterator.test_error()]
        for inference, error in zip(iterator.inferences, errors):
            inference_label = inference_labels[inference.label()]
            df.at[training_label, inference_label] = 100 * error

    fig, axes = plt.subplots(figsize=(TWO_COLUMNS_WIDTH, 0.5 * TWO_COLUMNS_WIDTH))
    fig.tight_layout()

    im = plt.imshow(df, norm=matplotlib.colors.LogNorm(), cmap="cividis")
    plt.yticks(np.arange(len(df.index)), df.index)
    plt.xticks(np.arange(len(df.columns)), df.columns)
    plt.xlabel("Inference", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Training", fontsize=AXIS_LABEL_FONT_SIZE)
    axes.xaxis.set_label_position("top")
    axes.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(axes.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
    cbar = fig.colorbar(im, ax=axes)
    cbar.ax.set_ylabel("Median error (%)", rotation=-90, va="bottom", fontsize=AXIS_LABEL_FONT_SIZE)
    annotate_heatmap(im, valfmt="{x:.1f}", textcolors=("white", "black"), size=TICKS_FONT_SIZE)
    plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    plt.savefig(
        "plotting/nonideality-agnosticism.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
