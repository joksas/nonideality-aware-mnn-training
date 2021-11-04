import copy

import crossbar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulations
from scipy.io import loadmat
from training import architecture
from training.iterator import (D2DLognormal, Inference, Iterator,
                               IVNonlinearity, StuckAtGMin, Training)

from . import utils


def iv_nonlinearity_training_curves(metric="error", training_idx=0):
    fig, axes = utils.fig_init(2, 0.55, fig_shape=(2, 3), sharex=True, sharey=True)

    iterators = simulations.iv_nonlinearity.get_iterators()
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx
    # Same training, different inference.
    iterators.insert(3, iterators[0])
    inference_idxs = [0, 0, 0, 1, 0, 0]

    for idx, (iterator, inference_idx) in enumerate(zip(iterators, inference_idxs)):
        i, j = np.unravel_index(idx, axes.shape)
        axis = axes[i, j]
        utils.plot_training_curves(
            fig, axis, iterator, subfigure_idx=idx, metric=metric, inference_idx=inference_idx
        )
        if i + 1 == axes.shape[0]:
            axis.set_xlabel(utils.axis_label("epoch"))
        if j == 0:
            axis.set_ylabel(utils.axis_label(metric))

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=axes.shape[1],
        bbox_to_anchor=(0.5, 1.03),
    )

    utils.save_fig(fig, f"iv-nonlinearity-training-{metric}")


def iv_nonlinearity_test(metric="error"):
    fig, axes = utils.fig_init(1, 0.8)

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

    plt.xlabel(utils.axis_label("power-consumption"))
    plt.ylabel(utils.axis_label(metric, prepend="test"))

    utils.save_fig(fig, f"iv-nonlinearity-test-{metric}")


def iv_nonlinearity_cnn_results(metric="error", training_idx=0):
    fig, axes = utils.fig_init(2, 1 / 3, fig_shape=(1, 3), sharey=True)

    colors = utils.color_dict()

    iterators = simulations.iv_nonlinearity_cnn.get_iterators()
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx

    axes[0].set_ylabel(utils.axis_label(metric))

    # Error curves.
    for axis, iterator in zip(axes, iterators):
        utils.plot_training_curves(fig, axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"))

    # Box plots.
    axis = axes[2]
    for idx, (iterator, color) in enumerate(zip(iterators, [colors["vermilion"], colors["blue"]])):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Nonideality-aware"])

    axis.set_xlabel("Training")

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=len(axes),
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, f"iv-nonlinearity-cnn-results-{metric}")


def d2d_uniformity_conductance_histograms(is_effective=False, include_regularised=False):
    num_rows = 2
    if include_regularised:
        num_rows = 3

    fig, axes = utils.fig_init(
        1, num_rows * 0.75, fig_shape=(num_rows, 1), sharex=True, sharey=True
    )

    iterators = simulations.d2d_asymmetry.get_iterators()
    if not include_regularised:
        iterators = iterators[:2]
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue", "bluish-green"]]

    for axis, iterator, color in zip(axes, iterators, colors):
        iterator.is_training = True
        model = architecture.get_model(iterator, custom_weights_path=iterator.weights_path())
        weights = model.layers[1].combined_weights()
        G, _ = crossbar.map.w_params_to_G(weights, iterator.training.G_min, iterator.training.G_max)
        G = 1000 * G
        if is_effective:
            values = G[:, ::2] - G[:, 1::2]
        else:
            values = G
        utils.add_histogram(axis, values, color=color)

        axis.set_ylabel(utils.axis_label("count"))

    if is_effective:
        label = utils.axis_label("conductance", prepend="effective")
        filename = "d2d-uniformity-G-eff-histograms"
    else:
        label = utils.axis_label("conductance")
        filename = "d2d-uniformity-G-histograms"
    if include_regularised:
        filename += "-regularised"
    axes[-1].set_xlabel(label)

    utils.save_fig(fig, filename)


def d2d_uniformity_pos_neg_conductance_scatterplots(combined=True):
    if combined:
        num_rows = 1
        height_frac = 1.0
    else:
        num_rows = 3
        height_frac = 2.5
    fig, axes = utils.fig_init(1, height_frac, fig_shape=(num_rows, 1), sharex=True, sharey=True)

    iterators = simulations.d2d_asymmetry.get_iterators()
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue", "bluish-green"]]

    for idx, (iterator, color) in enumerate(zip(iterators, colors)):
        if combined:
            axis = axes
        else:
            axis = axes[idx]
        iterator.is_training = True
        model = architecture.get_model(iterator, custom_weights_path=iterator.weights_path())
        weights = model.layers[1].combined_weights()
        G, _ = crossbar.map.w_params_to_G(weights, iterator.training.G_min, iterator.training.G_max)
        G = 1000 * G
        utils.plot_scatter(axis, G[:, ::2], G[:, 1::2], color)

        axis.set_ylabel(r"$G_{-}$ (mS)")

    if combined:
        axes.set_xlabel(r"$G_{+}$ (mS)")
    else:
        axes[-1].set_xlabel(r"$G_{+}$ (mS)")

    filename = "d2d-uniformity-G-scatter"
    if combined:
        filename += "-combined"
    utils.save_fig(fig, filename)


def d2d_uniformity_pos_neg_conductance_histograms():
    fig, axes = utils.fig_init(1, 1.5, fig_shape=(2, 1), sharex=True, sharey=True)

    iterators = simulations.d2d_asymmetry.get_iterators()

    for axis, iterator in zip(axes, iterators):
        iterator.is_training = True
        model = architecture.get_model(iterator, custom_weights_path=iterator.weights_path())
        weights = model.layers[1].combined_weights()
        G, _ = crossbar.map.w_params_to_G(weights, iterator.training.G_min, iterator.training.G_max)
        G = 1000 * G
        utils.add_histogram(
            axis,
            G[:, ::2],
            color=utils.color_dict()["bluish-green"],
            alpha=0.5,
        )
        utils.add_histogram(
            axis,
            G[:, 1::2],
            color=utils.color_dict()["reddish-purple"],
            alpha=0.5,
        )
        axis.set_ylabel(utils.axis_label("count"))

    axes[1].set_xlabel(utils.axis_label("conductance"))

    utils.add_legend(
        fig,
        [r"$G_+$", r"$G_-$"],
        ncol=len(axes),
        bbox_to_anchor=(0.55, 1.0),
    )

    utils.save_fig(fig, "d2d-uniformity-G-pos-neg-histograms")


def d2d_uniformity_results(metric="error", training_idx=0):
    fig, axes = utils.fig_init(2, 1 / 3, fig_shape=(1, 3), sharey=True)

    colors = utils.color_dict()

    iterators = simulations.d2d_asymmetry.get_iterators()[:2]
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx

    axes[0].set_ylabel(utils.axis_label(metric))

    # Error curves.
    for axis, iterator in zip(axes, iterators):
        utils.plot_training_curves(fig, axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"))

    # Box plots.
    axis = axes[2]
    for idx, (iterator, color) in enumerate(zip(iterators, [colors["vermilion"], colors["blue"]])):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["High", "Low"])
    axis.set_xlabel(utils.axis_label("d2d-uniformity"))

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=len(axes),
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, f"d2d-uniformity-results-{metric}")


def iv_nonlinearity_and_stuck_results(metric="error", training_idx=0):
    fig, axes = utils.fig_init(2, 1 / 3, fig_shape=(1, 3), sharey=True)

    iterators = simulations.iv_nonlinearity_and_stuck.get_iterators()
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx

    axes[0].set_ylabel(utils.axis_label(metric))

    # Curves
    for idx, (iterator, axis) in enumerate(zip(iterators, axes)):
        utils.plot_training_curves(fig, axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"))

    # Box plots
    axis = axes[-1]
    boxplots = []
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    for idx, (iterator, color) in enumerate(zip(iterators, colors)):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Nonideality-aware"])
    axis.set_xlabel(utils.axis_label("training"))

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=len(axes),
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, f"iv-nonlinearity-and-stuck-results-{metric}")


def checkpoint_comparison_boxplots(metric="error", training_idx=0):
    fig, axes = utils.fig_init(2, 1 / 3, fig_shape=(1, 3), sharey=True)

    iterators = simulations.checkpoint_comparison.get_iterators()
    for i in range(len(iterators)):
        iterators[i].training.repeat_idx = training_idx

    axes[0].set_ylabel(utils.axis_label(metric))

    # Curves
    for idx, (iterator, axis) in enumerate(zip(iterators, axes)):
        utils.plot_training_curves(fig, axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"))

    # Box plots
    axis = axes[-1]
    boxplots = []
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    for idx, (iterator, color) in enumerate(zip(iterators, colors)):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Memristive"])
    axis.set_xlabel(utils.axis_label("checkpoint"))

    utils.add_legend(
        fig,
        ["Training", "Validation", "Test (nonideal)"],
        ncol=len(axes),
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, f"checkpoint-results-{metric}")


def nonideality_agnosticism_heatmap(metric="error"):
    training_labels = {
        "nonreg__64__none_none__ideal": "Ideal",
        "nonreg__64__0.000997_0.00351__IVNL:2.13_0.095": r"Low $I$-$V$ nonlin.",
        "reg__64__0.000997_0.00351__IVNL:2.13_0.095": r"Low $I$-$V$ nonlin. (reg.)",
        "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369": r"High $I$-$V$ nonlin.",
        "reg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369": r"High $I$-$V$ nonlin. (reg.)",
        "nonreg__64__7.72e-07_2.73e-06__StuckMin:0.05": r"Stuck at $G_\mathrm{off}$",
        "nonreg__64__0.000997_0.00351__D2DLN:0.25_0.25": "More uniform D2D var.",
        "nonreg__64__0.000997_0.00351__D2DLN:0.05_0.5": "Less uniform D2D var.",
        "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369+StuckMax:0.05": r"High $I$-$V$ nonlin. + stuck at $G_\mathrm{on}$",
    }
    inference_labels = {
        "none_none__ideal": training_labels["nonreg__64__none_none__ideal"],
        "0.000997_0.00351__IVNL:2.13_0.095": training_labels[
            "nonreg__64__0.000997_0.00351__IVNL:2.13_0.095"
        ],
        "7.72e-07_2.73e-06__IVNL:2.99_0.369": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369"
        ],
        "7.72e-07_2.73e-06__StuckMin:0.05": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__StuckMin:0.05"
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
        ys = [
            iterator.test_metric(metric, inference_idx=idx)
            for idx in range(len(iterator.inferences))
        ]
        for inference, y in zip(iterator.inferences, ys):
            inference_label = inference_labels[inference.label()]
            df.at[training_label, inference_label] = np.median(y)

    fig, axes = utils.fig_init(2, 0.5)

    utils.add_heatmap(fig, axes, df, x_ticks=df.columns, y_ticks=df.index, metric=metric)

    axes.set_ylabel("Training")
    axes.set_xlabel("Inference")

    utils.save_fig(fig, f"nonideality-agnosticism-{metric}")


def iv_curves_low_high(data_filepath):
    # plt.rc("font", **{"size": "1"})
    fig, axes = utils.fig_init(2, 0.4, fig_shape=(1, 2), sharex=True)

    low_n_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    high_n_idxs = [
        40,
        31,
        47,
        36,
        51,
        50,
        30,
        45,
        39,
        52,
        33,
        35,
        34,
        41,
        48,
        43,
        38,
        42,
        37,
        44,
        46,
    ]

    data = loadmat(data_filepath)["data"]
    data = np.flip(data, axis=2)

    N = 1000
    palette = plt.cm.inferno(np.linspace(0, 1, N))

    for idx, (axis, state_idxs) in enumerate(zip(axes, [low_n_idxs, high_n_idxs])):
        min_voltage = np.inf
        max_voltage = -np.inf
        min_current = np.inf
        max_current = -np.inf

        for state_idx in state_idxs:
            voltages = data[:101, 1, state_idx]
            currents = data[:101, 0, state_idx]
            if np.max(voltages) > max_voltage:
                max_voltage = np.max(voltages)
            if np.min(voltages) < max_voltage:
                min_voltage = np.min(voltages)
            if np.max(currents) > max_current:
                max_current = np.max(currents)
            if np.min(currents) < max_current:
                min_current = np.min(currents)

            n = currents[100] / currents[50]
            palette_idx = int(np.floor(N * (n - 2) / 2))
            axis.plot(
                voltages, currents, color=palette[palette_idx], linewidth=utils.Config.LINEWIDTH
            )

        axis.set_xlim([min_voltage, max_voltage])
        axis.set_ylim(bottom=0)
        axis.set_xlabel(utils.axis_label("voltage"))
        utils.add_subfigure_label(fig, axis, idx)
        axis.ticklabel_format(axis="y", scilimits=(-1, 1))
        axis.tick_params(axis="both", which="both", labelsize=utils.Config.TICKS_FONT_SIZE)
        axis.yaxis.get_offset_text().set_fontsize(utils.Config.TICKS_FONT_SIZE)

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=2, vmax=4))
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label(
        label=utils.axis_label("nonlinearity-parameter"),
        fontsize=utils.Config.AXIS_LABEL_FONT_SIZE,
        rotation=-90,
        va="bottom",
    )
    cbar.ax.tick_params(axis="both", which="both", labelsize=utils.Config.TICKS_FONT_SIZE)

    axes[0].set_ylabel(utils.axis_label("current"))

    utils.save_fig(fig, "SiO_x-I-V-curves")
