import numpy as np
import scipy.constants as const

from awarememristor import simulations
from awarememristor.crossbar import nonidealities
from awarememristor.plotting import utils


def all_iv_curves_full_range():
    fig, axes = utils.fig_init(2, 0.6, fig_shape=(1, 1))

    data = simulations.data.load_SiO_x_multistate()
    voltages, currents = simulations.data.all_SiO_x_curves(data, max_voltage=5.0, clean_data=False)

    colors = utils.color_list()[:-1]
    num_colors = len(colors)
    for idx in range(voltages.shape[0]):
        voltage_curve = voltages[idx, :]
        current_curve = currents[idx, :]
        color_idx = idx % num_colors
        axes.plot(
            voltage_curve,
            current_curve,
            linewidth=utils.Config.LINEWIDTH,
            color=colors[color_idx],
        )

    axes.set_xlim(left=0.0)
    axes.set_xlabel(utils.axis_label("voltage"))
    axes.yaxis.get_offset_text().set_fontsize(utils.Config.TICKS_FONT_SIZE)

    axes.set_ylabel(utils.axis_label("current"))
    axes.set_yscale("log")

    utils.save_fig(fig, "all-SiO_x-IV-curves-full-range", is_supporting=True)


def switching():
    fig, axes = utils.fig_init(2, 0.5, fig_shape=(1, 1))

    data = simulations.data.load_SiO_x_switching()
    data[:, 0, :] = np.abs(data[:, 0, :])

    colors = [utils.color_dict()[color_name] for color_name in ["blue", "orange"]]
    labels = ["SET", "RESET"]
    labels_x = [0.15, 1 - 0.15]
    for idx, (color, label, label_x) in enumerate(zip(colors, labels, labels_x)):
        voltage_curve = data[:, 1, idx]
        current_curve = np.abs(data[:, 0, idx])
        line = axes.plot(
            voltage_curve,
            current_curve,
            linewidth=utils.Config.LINEWIDTH,
            color=color,
        )
        utils.add_arrow(line[0], 60)
        utils.add_arrow(line[0], -60)
        utils.add_text(
            axes,
            label,
            (label_x, 0.88),
            fontsize=utils.Config.TEXT_LABEL_SIZE,
            color=color,
        )

    axes.set_xlabel(utils.axis_label("voltage"))

    axes.set_ylabel(utils.axis_label("current", prepend="absolute"))
    axes.set_ylim(bottom=5e-8, top=5e-3)
    axes.set_yscale("log")

    utils.save_fig(fig, "SiO_x-switching", is_supporting=True)


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
                axis,
                iterator,
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


def high_d2d_training(metric="error"):
    iterators = simulations.high_d2d.get_iterators()

    _training_curves_multiple_panels(
        (2, 3),
        0.45,
        (1, 2),
        iterators,
        metric,
        "high-d2d",
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
        "high-iv-nonlinearity-and-stuck-on",
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
        y_lim=95,
    )


def pf_plots():
    fig, axes = utils.fig_init(2, 1.0, fig_shape=(7, 7), sharex=True, no_panels=True)

    exp_data = simulations.data.load_SiO_x_multistate()
    V_full, I_full = simulations.data.all_SiO_x_curves(exp_data, clean_data=True)
    (
        resistances,
        _,
        _,
        V_full,
        I_full,
    ) = simulations.data.pf_relationship(V_full, I_full)
    R_0 = const.physical_constants["inverse of conductance quantum"][0]
    # Separate data into before and after the conductance quantum.
    sep_idx = np.searchsorted(resistances, R_0)
    colors = utils.color_dict()

    axis_idx = 0
    for is_high_resistance in [False, True]:
        if is_high_resistance:
            idxs = np.arange(sep_idx, len(resistances))
            color = colors["reddish-purple"]
        else:
            idxs = np.arange(sep_idx)
            color = colors["blue"]

        V = V_full[idxs]
        I = I_full[idxs]

        for idx in range(V.shape[0]):
            axis = axes[np.unravel_index(axis_idx, axes.shape)]
            voltage_curve = V[idx, 21:]
            current_curve = I[idx, 21:]
            axis.plot(
                np.sqrt(voltage_curve),
                np.log(current_curve / voltage_curve),
                linewidth=utils.Config.LINEWIDTH,
                color=color,
            )
            axis_idx += 1

    for axis_idx in range(axes.shape[1]):
        axis = axes[-1, axis_idx]
        axis.set_xlabel(utils.axis_label("sqrt(V)"))
    for axis_idx in range(axes.shape[0]):
        axis = axes[axis_idx, 0]
        axis.set_ylabel(utils.axis_label("ln(I/V)"))

    utils.save_fig(fig, "pf-plots", is_supporting=True)


def pf_fits():
    fig, axes = utils.fig_init(2, 1.0, fig_shape=(7, 7), sharex=True, no_panels=True)

    exp_data = simulations.data.load_SiO_x_multistate()
    V_full, I_full = simulations.data.all_SiO_x_curves(exp_data, clean_data=True)
    (
        resistances,
        c_full,
        d_times_perm_full,
        V_full,
        I_full,
    ) = simulations.data.pf_relationship(V_full, I_full)
    R_0 = const.physical_constants["inverse of conductance quantum"][0]
    # Separate data into before and after the conductance quantum.
    sep_idx = np.searchsorted(resistances, R_0)
    colors = utils.color_dict()

    axis_idx = 0
    for is_high_resistance in [False, True]:
        if is_high_resistance:
            idxs = np.arange(sep_idx, len(resistances))
            color = colors["reddish-purple"]
        else:
            idxs = np.arange(sep_idx)
            color = colors["blue"]

        V = V_full[idxs]
        I = I_full[idxs]
        c = c_full[idxs]
        d_times_perm = d_times_perm_full[idxs]

        for idx in range(V.shape[0]):
            axis = axes[np.unravel_index(axis_idx, axes.shape)]
            voltage_curve = V[idx, :]
            current_curve = I[idx, :]
            axis.plot(
                voltage_curve,
                current_curve,
                linewidth=utils.Config.LINEWIDTH,
                color=color,
            )
            fit = nonidealities.IVNonlinearityPF.model(voltage_curve, c[idx], d_times_perm[idx])
            axis.plot(
                voltage_curve,
                fit,
                linewidth=utils.Config.LINEWIDTH,
                color=colors["black"],
                linestyle="dotted",
            )
            axis.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
            axis.yaxis.get_offset_text().set_fontsize(utils.Config.TICKS_FONT_SIZE)
            axis_idx += 1

    axes[-1, 3].set_xlabel(utils.axis_label("voltage"))
    axes[3, 0].set_ylabel(utils.axis_label("current"))

    utils.save_fig(fig, "pf-fits", is_supporting=True)
