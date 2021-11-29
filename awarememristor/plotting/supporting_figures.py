import matplotlib.pyplot as plt
import numpy as np

from awarememristor import simulations
from awarememristor.plotting import utils


def iv_nonlinearity_training(metric="error"):
    iterators = simulations.iv_nonlinearity.get_iterators()
    # Same training, different inference.
    iterators.insert(3, iterators[0])
    inference_idxs = [0, 0, 0, 1, 0, 0]

    _training_curves_multiple_panels(
        2,
        0.55,
        (2, 3),
        iterators,
        metric,
        "iv-nonlinearity",
        inference_idxs=inference_idxs,
    )


def weight_implementation_standard_weights_training(metric="error"):
    iterators = simulations.weight_implementation.get_nonideal_iterators()[:4]

    _training_curves_multiple_panels(
        (2, 3),
        0.77,
        (2, 2),
        iterators,
        metric,
        "weight-implementation-standard-weights",
    )


def weight_implementation_double_weights_training(metric="error"):
    iterators = [
        simulations.weight_implementation.get_ideal_iterator()
    ] + simulations.weight_implementation.get_nonideal_iterators()[4:]
    # Same training, different inference.
    iterators.insert(3, iterators[0])
    inference_idxs = [0, 0, 0, 1, 0, 0]

    _training_curves_multiple_panels(
        2,
        0.55,
        (2, 3),
        iterators,
        metric,
        "weight-implementation-double-weights",
        inference_idxs=inference_idxs,
        y_lim=95,
    )


def memristive_validation_training(metric="error"):
    iterators = simulations.memristive_validation.get_iterators()

    _training_curves_multiple_panels(
        2,
        0.3,
        (1, 3),
        iterators,
        metric,
        "memristive-validation",
        y_lim=95,
    )


def stuck_off_training(metric="error"):
    iterators = simulations.stuck_off.get_iterators()

    _training_curves_multiple_panels(
        (2, 3),
        0.45,
        (1, 2),
        iterators,
        metric,
        "stuck-off",
    )


def high_iv_nonlinearity_and_stuck_on_training(metric="error"):
    iterators = simulations.iv_nonlinearity_and_stuck_on.get_iterators()

    _training_curves_multiple_panels(
        (2, 3),
        0.45,
        (1, 2),
        iterators,
        metric,
        "iv-nonlinearity-and-stuck-on",
    )


def stuck_distribution_training(metric="error"):
    iterators = simulations.stuck_distribution.get_iterators()

    _training_curves_multiple_panels(
        (2, 3),
        0.45,
        (1, 2),
        iterators,
        metric,
        "stuck-distribution",
    )


def _training_curves_multiple_panels(
    width_num_cols,
    height_frac,
    fig_shape,
    iterators,
    metric,
    figure_name,
    inference_idxs=None,
    y_lim=None,
):
    fig, axes = utils.fig_init(
        width_num_cols, height_frac, fig_shape=fig_shape, sharex=True, sharey=True
    )
    if inference_idxs is None:
        inference_idxs = [0 for _ in range(len(iterators))]

    for training_idx, linestyle in enumerate(utils.get_linestyles()):
        for i in range(len(iterators)):
            iterators[i].training.repeat_idx = training_idx

        for idx, (iterator, inference_idx) in enumerate(zip(iterators, inference_idxs)):
            if len(axes.shape) == 1:
                axis = axes[idx]
            else:
                i, j = np.unravel_index(idx, axes.shape)
                axis = axes[i, j]
            utils.plot_training_curves(
                fig,
                axis,
                iterator,
                subfigure_idx=idx,
                metric=metric,
                inference_idx=inference_idx,
                linestyle=linestyle,
                is_many=True,
            )
            if len(axes.shape) == 1:
                axis.set_xlabel(utils.axis_label("epoch"))
                if idx == 0:
                    axis.set_ylabel(utils.axis_label(metric))
            else:
                if i + 1 == axes.shape[0]:
                    axis.set_xlabel(utils.axis_label("epoch"))
                if j == 0:
                    axis.set_ylabel(utils.axis_label(metric))
            if y_lim is not None:
                axis.set_ylim(top=y_lim)

    utils.add_legend(
        fig,
        labels=["Training", "Validation", "Test (nonideal)"],
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
    )

    utils.save_fig(fig, f"{figure_name}-training", is_supporting=True, metric=metric)


def iv_curves_all():
    fig, axes = utils.fig_init(1, 0.8, fig_shape=(1, 1))

    data = simulations.data.load_SiO_x()
    voltages, currents = simulations.data.all_SiO_x_curves(data)

    N = 1000
    palette = plt.cm.inferno(np.linspace(0, 1, N))

    min_voltage, max_voltage = 0.0, 0.5

    for idx in range(voltages.shape[0]):
        voltage_curve = voltages[idx, :]
        current_curve = currents[idx, :]
        n = simulations.data.nonlinearity_parameter(current_curve)
        palette_idx = int(np.floor(N * (n - 2) / 2))
        axes.plot(
            voltage_curve,
            current_curve,
            color=palette[palette_idx],
            linewidth=utils.Config.LINEWIDTH,
        )

    axes.set_xlim([min_voltage, max_voltage])
    axes.set_xlabel(utils.axis_label("voltage"))
    axes.yaxis.get_offset_text().set_fontsize(utils.Config.TICKS_FONT_SIZE)

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=2, vmax=4))
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label(
        label=utils.axis_label("nonlinearity-parameter"),
        fontsize=utils.Config.AXIS_LABEL_FONT_SIZE,
        rotation=-90,
        va="bottom",
    )
    cbar.ax.tick_params(axis="both", which="both", labelsize=utils.Config.TICKS_FONT_SIZE)

    axes.set_ylabel(utils.axis_label("current"))
    axes.set_yscale("log")

    utils.save_fig(fig, "SiO_x-IV-curves-all", is_supporting=True)
