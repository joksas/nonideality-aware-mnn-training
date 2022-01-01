import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.lines import Line2D

from awarememristor import crossbar, simulations
from awarememristor.crossbar.nonidealities import StuckDistribution
from awarememristor.plotting import utils
from awarememristor.training import architecture

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)


def _SiO_x_panels(fig, axes):
    data = simulations.data.load_SiO_x_multistate()

    N = 1000
    palette = plt.cm.inferno(np.linspace(0, 1, N))
    min_voltage, max_voltage = 0.0, 0.5

    curves = simulations.data.low_high_n_SiO_x_curves(data)
    for axis, (voltages, currents) in zip(axes, curves):
        for idx in range(voltages.shape[0]):
            voltage_curve = voltages[idx, :]
            current_curve = currents[idx, :]
            n = simulations.data.nonlinearity_parameter(current_curve)
            palette_idx = int(np.floor(N * (n - 2) / 2))
            axis.plot(
                voltage_curve,
                current_curve,
                color=palette[palette_idx],
                linewidth=utils.Config.LINEWIDTH,
            )

        axis.set_xlim([min_voltage, max_voltage])
        axis.set_ylim(bottom=0)
        axis.set_xlabel(utils.axis_label("voltage"))
        axis.ticklabel_format(axis="y", scilimits=(-1, 1))
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


def _HfO2_panels(fig, axes):
    data = simulations.data.load_Ta_HfO2()
    G_min, G_max = simulations.data.extract_G_off_and_G_on(data)
    vals, p = simulations.data.extract_stuck(data, G_min, G_max)
    median_range = G_max - G_min
    colors = utils.color_dict()

    axis = axes[0]
    shape = data.shape
    num_pulses = shape[0] * shape[1]
    num_bl = shape[2]
    num_wl = shape[3]
    pulsing_step_size = 10
    random_proportion = 0.01
    x = [i + 1 for i in range(0, num_pulses, pulsing_step_size)]
    data = np.reshape(data, (num_pulses, num_bl, num_wl))
    num_devices = num_wl * num_bl
    num_reduced_devices = int(np.around(random_proportion * num_devices))
    np.random.seed(0)
    random_idxs = np.random.choice(num_devices, num_reduced_devices)
    bl_idxs, wl_idxs = np.unravel_index(random_idxs, (num_bl, num_wl))
    for bl_idx, wl_idx in zip(bl_idxs, wl_idxs):
        curve_data = data[:, bl_idx, wl_idx]
        if np.max(curve_data) - np.min(curve_data) < simulations.data.stuck_device_threshold(
            median_range
        ):
            color = colors["vermilion"]
        else:
            color = colors["bluish-green"]
        y = curve_data[::pulsing_step_size]
        axis.plot(x, 1000 * y, color=color, lw=utils.Config.LINEWIDTH / 2, alpha=1 / 2)

    for G in [G_min, G_max]:
        axis.axhline(
            1000 * G,
            0,
            1,
            color=colors["blue"],
            lw=utils.Config.LINEWIDTH,
            linestyle="dashed",
            zorder=10,
        )

    axis.set_xlabel(utils.axis_label("pulse-number"))
    axis.set_ylabel(utils.axis_label("conductance", unit_prefix="m"))

    axis.set_xlim([0, x[-1]])
    axis.set_ylim(bottom=0.0)

    handles = [
        Line2D([0], [0], color=colors["vermilion"], label="Stuck devices"),
        Line2D([0], [0], color=colors["bluish-green"], label="Other devices"),
        Line2D(
            [0],
            [0],
            color=colors["blue"],
            linestyle="dashed",
            label=r"$G_\mathrm{off}, G_\mathrm{on}$",
        ),
    ]

    # Distribution
    axis = axes[1]
    distribution = StuckDistribution(vals, p).distribution
    x = np.linspace(0.0, 1.5e-3, int(1e4))
    y = distribution.prob(x)
    y = y / 1000
    x = 1000 * x

    axis.plot(y, x, lw=0.5, color=colors["vermilion"])
    axis.scatter(
        np.zeros_like(vals),
        [1000 * val for val in vals],
        marker="_",
        alpha=0.1,
        lw=utils.Config.LINEWIDTH / 2,
        color=colors["vermilion"],
    )
    for G in [G_min, G_max]:
        axis.axhline(
            1000 * G,
            0,
            1,
            color=colors["blue"],
            lw=utils.Config.LINEWIDTH,
            linestyle="dashed",
            zorder=10,
        )

    axis.set_xlabel(r"Probability density ($\mathrm{mS}^{-1}$)")

    axis.set_xlim(left=0.0)

    utils.add_legend(
        fig,
        ncol=3,
        bbox_to_anchor=(0.5, 0.515),
        handles=handles,
    )


def experimental_data():
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.08, 1.1])

    gs_top = gs[0].subgridspec(1, 2, wspace=0.03)
    gs_bottom = gs[2].subgridspec(1, 2, wspace=0.03)

    subplots = list(gs_top) + list(gs_bottom)
    for subplot in subplots:
        fig.add_subplot(subplot)

    fig, axes = utils.fig_init(2, 0.9, custom_fig=fig)

    axes[1].sharex(axes[0])
    axes[3].sharey(axes[2])
    axes[3].label_outer()

    _SiO_x_panels(fig, axes[[0, 1]])
    _HfO2_panels(fig, axes[[2, 3]])

    utils.save_fig(fig, "experimental-data")


def iv_nonlinearity_training(metric="error"):
    fig, axes = utils.fig_init(2, 0.55, fig_shape=(2, 3), sharex=True, sharey=True)

    iterators = simulations.iv_nonlinearity.get_iterators()
    # Same training, different inference.
    iterators.insert(3, iterators[0])
    inference_idxs = [0, 0, 0, 1, 0, 0]

    for idx, (iterator, inference_idx) in enumerate(zip(iterators, inference_idxs)):
        i, j = np.unravel_index(idx, axes.shape)
        axis = axes[i, j]
        utils.plot_training_curves(axis, iterator, metric=metric, inference_idx=inference_idx)
        if i + 1 == axes.shape[0]:
            axis.set_xlabel(utils.axis_label("epoch"))
        if j == 0:
            axis.set_ylabel(utils.axis_label(metric))

    utils.add_legend(
        fig,
        labels=["Training", "Validation", "Test (nonideal)"],
        ncol=axes.shape[1],
        bbox_to_anchor=(0.5, 1.03),
    )

    utils.save_fig(fig, "iv-nonlinearity-training", metric=metric)


def iv_nonlinearity_inference(metric="error"):
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
        boxplot = utils.plot_boxplot(
            axes, y, color, metric=metric, x=avg_power, is_x_log=True, linear_width=0.2
        )
        boxplots.append(boxplot)

    utils.add_boxplot_legend(
        axes, boxplots, ["Standard", "Nonideality-aware", "Nonideality-aware (regularised)"]
    )

    plt.xlabel(utils.axis_label("power-consumption"))
    plt.ylabel(utils.axis_label(metric))

    utils.save_fig(fig, "iv-nonlinearity-inference", metric=metric)


def iv_nonlinearity_cnn(metric="error"):
    fig, axes = utils.fig_init(2, 1 / 3, fig_shape=(1, 3), sharey=True)

    colors = utils.color_dict()

    iterators = simulations.iv_nonlinearity_cnn.get_iterators()

    axes[0].set_ylabel(utils.axis_label(metric))

    # Error curves.
    for axis, iterator in zip(axes, iterators):
        utils.plot_training_curves(axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"))

    # Box plots.
    axis = axes[2]
    for idx, (iterator, color) in enumerate(zip(iterators, [colors["vermilion"], colors["blue"]])):
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Nonideality-aware"])

    axis.set_xlabel("Training")
    axis.set_ylim(top=95.0)

    utils.add_legend(
        fig,
        labels=["Training", "Validation", "Test (nonideal)"],
        ncol=len(axes),
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, "iv-nonlinearity-cnn", metric=metric)


def weight_implementation(metric="error"):
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.8])

    gs_top = gs[0].subgridspec(2, 4)
    gs_bottom = gs[1].subgridspec(1, 2)

    subplots = list(gs_top) + list(gs_bottom)
    for subplot in subplots:
        fig.add_subplot(subplot)

    fig, axes = utils.fig_init(2, 1.0, custom_fig=fig)

    for axis in axes[:8]:
        axis.sharex(axes[4])
        axis.sharey(axes[0])
        axis.label_outer()
        axis.set_aspect("equal", adjustable="box")

    iterators = simulations.weight_implementation.get_iterators()[1:]
    colors = [
        utils.color_dict()[key] for key in ["reddish-purple", "vermilion", "blue", "bluish-green"]
    ]

    temp_iterators = [iterators[idx] for idx in [0, 1, 4, 5, 2, 3, 6, 7]]
    for idx, (axis, iterator, color) in enumerate(zip(axes, temp_iterators, colors + colors)):
        iterator.is_training = False
        iterator.inference_idx = 0
        model = architecture.get_model(iterator, custom_weights_path=iterator.weights_path())
        weights = model.layers[1].combined_weights()

        inference = iterator.current_stage()
        G_off = inference.G_off
        G_on = inference.G_on

        if iterator.training.uses_double_weights():
            G, _ = crossbar.map.double_w_to_G(weights, G_off, G_on)
        else:
            G, _ = crossbar.map.w_to_G(weights, G_off, G_on, mapping_rule=inference.mapping_rule)

        G = 1e6 * G
        utils.plot_scatter(axis, G[:, ::2], G[:, 1::2], color, random_proportion=0.1)

        axis.xaxis.set_ticks(np.arange(1.0, 3.0, 0.5))
        axis.yaxis.set_ticks(np.arange(1.0, 3.0, 0.5))

        if idx > 3:
            axis.set_xlabel(utils.axis_label("g-plus", unit_prefix="\\textmu{}"))
        if idx in [0, 4]:
            axis.set_ylabel(utils.axis_label("g-minus", unit_prefix="$\\textmu{}$"))

    for iterator_idxs, axis in zip([[0, 1, 4, 5], [2, 3, 6, 7]], axes[-2:]):
        for iterator_idx, color in zip(iterator_idxs, colors):
            iterator = iterators[iterator_idx]
            avg_power = iterator.test_metric("avg_power")
            y = iterator.test_metric(metric)
            utils.plot_boxplot(axis, y, color, x=1000 * avg_power, metric=metric, linear_width=0.2)
            axis.set_xlabel(utils.axis_label("power-consumption", unit_prefix="m"))

    axes[-2].set_ylabel(utils.axis_label(metric))
    axes[-1].sharey(axes[-2])
    axes[-1].sharex(axes[-2])
    axes[-1].label_outer()

    utils.save_fig(fig, "weight-implementation", metric=metric)


def memristive_validation(metric="error"):
    fig, axes = utils.fig_init(2, 1 / 3, fig_shape=(1, 3), sharey=True)

    iterator = simulations.memristive_validation.get_nonideal_iterators()[0]

    axes[0].set_ylabel(utils.axis_label(metric))

    # Curves
    for idx, (standard_mode, axis) in enumerate(zip([True, False], axes)):
        iterator.training.is_standard_validation_mode = standard_mode
        utils.plot_training_curves(axis, iterator, metric=metric)
        axis.set_xlabel(utils.axis_label("epoch"))

    # Box plots
    axis = axes[-1]
    colors = [utils.color_dict()[key] for key in ["vermilion", "blue"]]

    for idx, (standard_mode, color) in enumerate(zip([True, False], colors)):
        iterator.training.is_standard_validation_mode = standard_mode
        y = iterator.test_metric(metric)
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Memristive"])
    axis.set_xlabel(utils.axis_label("checkpoint"))

    utils.add_legend(
        fig,
        labels=["Training", "Validation", "Test (nonideal)"],
        ncol=len(axes),
        bbox_to_anchor=(0.35, 1.05),
    )

    utils.save_fig(fig, "memristive-validation", metric=metric)


def nonideality_agnosticism(metric: str = "error", norm_rows=True, include_val_label=False):
    training_labels = {
        "nonreg__64__none_none__ideal": "Ideal",
        "nonreg__64__0.000997_0.00351__IVNL:2.13_0.0953": r"Low $I$-$V$ nonlin. [$\mathrm{SiO}_x$]",
        "reg__64__0.000997_0.00351__IVNL:2.13_0.0953": r"Low $I$-$V$ nonlin. [$\mathrm{SiO}_x$] (reg.)",
        "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369": r"High $I$-$V$ nonlin. [$\mathrm{SiO}_x$]",
        "reg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369": r"High $I$-$V$ nonlin. [$\mathrm{SiO}_x$] (reg.)",
        "nonreg__64__7.72e-07_2.73e-06__StuckOff:0.05": r"Stuck at $G_\mathrm{off}$",
        "nonreg__64__4.36e-05_0.000978__StuckDistr:0.101_1.77e-05": r"Stuck [$\mathrm{Ta/HfO}_2$]",
        "nonreg__64__7.72e-07_2.73e-06__D2DLN:0.25_0.25": "More uniform D2D var.",
        "reg__64__7.72e-07_2.73e-06__D2DLN:0.25_0.25": "More uniform D2D var. (reg.)",
        "nonreg__64__7.72e-07_2.73e-06__D2DLN:0.05_0.5": "Less uniform D2D var.",
        "reg__64__7.72e-07_2.73e-06__D2DLN:0.05_0.5": "Less uniform D2D var. (reg.)",
        "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369+StuckOn:0.05": r"High $I$-$V$ nonlin. [$\mathrm{SiO}_x$] + stuck at $G_\mathrm{on}$",
        "nonreg__64__7.72e-07_2.73e-06__D2DLN:0.5_0.5": "High D2D var.",
    }
    inference_labels = {
        "none_none__ideal": training_labels["nonreg__64__none_none__ideal"],
        "0.000997_0.00351__IVNL:2.13_0.0953": training_labels[
            "nonreg__64__0.000997_0.00351__IVNL:2.13_0.0953"
        ],
        "7.72e-07_2.73e-06__IVNL:2.99_0.369": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369"
        ],
        "7.72e-07_2.73e-06__StuckOff:0.05": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__StuckOff:0.05"
        ],
        "4.36e-05_0.000978__StuckDistr:0.101_1.77e-05": training_labels[
            "nonreg__64__4.36e-05_0.000978__StuckDistr:0.101_1.77e-05"
        ],
        "7.72e-07_2.73e-06__D2DLN:0.25_0.25": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__D2DLN:0.25_0.25"
        ],
        "7.72e-07_2.73e-06__D2DLN:0.05_0.5": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__D2DLN:0.05_0.5"
        ],
        "7.72e-07_2.73e-06__IVNL:2.99_0.369+StuckOn:0.05": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__IVNL:2.99_0.369+StuckOn:0.05"
        ],
        "7.72e-07_2.73e-06__D2DLN:0.5_0.5": training_labels[
            "nonreg__64__7.72e-07_2.73e-06__D2DLN:0.5_0.5"
        ],
    }
    df = pd.DataFrame(
        columns=[training_labels[key] for key in training_labels],
        index=[inference_labels[key] for key in inference_labels],
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
            df.at[inference_label, training_label] = np.median(y)

    fig, axes = utils.fig_init(2, 0.5)

    filename = "nonideality-agnosticism"
    if not norm_rows:
        filename += "-not-norm"
    if include_val_label:
        filename += "-with-val-label"

    utils.add_heatmap(
        fig, axes, df, x_ticks=df.columns, y_ticks=df.index, metric=metric, norm_rows=norm_rows
    )

    axes.set_ylabel(utils.axis_label("inference"))
    axes.set_xlabel(utils.axis_label("training"))

    if include_val_label:
        axes.text(
            1.05,
            0.5,
            utils.axis_label("error", prepend="median"),
            horizontalalignment="center",
            verticalalignment="center",
            rotation=-90,
            transform=axes.transAxes,
            fontsize=utils.Config.AXIS_LABEL_FONT_SIZE,
        )

    utils.save_fig(fig, filename, metric=metric)
