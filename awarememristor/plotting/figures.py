import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
import scipy.stats as stats
from matplotlib.lines import Line2D

from awarememristor import crossbar, simulations
from awarememristor.crossbar.nonidealities import StuckDistribution
from awarememristor.plotting import utils
from awarememristor.training import architecture

logging.getLogger().setLevel(logging.INFO)

plt.rcParams.update(
    {
        "font.sans-serif": "Fira Sans",
        "font.weight": "light",
    }
)


def SiO_x():
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 1)

    gs_top = gs[0].subgridspec(1, 2, wspace=0.03)

    subplots = list(gs_top)
    for subplot in subplots:
        fig.add_subplot(subplot)

    fig, axes = utils.fig_init(2, 0.4, custom_fig=fig)

    axes[1].sharex(axes[0])

    N = 1000
    v_min = 1.0
    v_max = 1.5
    palette = plt.cm.inferno(np.linspace(0, 1, N))
    min_voltage, max_voltage = 0.0, 0.5

    exp_data = simulations.data.load_SiO_x_multistate()
    V_full, I_full = simulations.data.all_SiO_x_curves(exp_data, clean_data=True)
    (
        resistances,
        _,
        _,
        V_full,
        I_full,
    ) = simulations.data.pf_relationship(V_full, I_full)
    low_idxs, high_idxs = simulations.data.edge_state_idxs(
        resistances, simulations.data.SiO_x_G_on_G_off_ratio()
    )

    logging.info(
        "Range of low-resistance states: %.4g-%.4g ohms",
        resistances[low_idxs[0]],
        resistances[low_idxs[-1]],
    )
    logging.info(
        "Range of high-resistance states: %.4g-%.4g ohms",
        resistances[high_idxs[0]],
        resistances[high_idxs[-1]],
    )
    nonlinearities = []
    for i in range(V_full.shape[0]):
        voltage_curve = V_full[i, :]
        current_curve = I_full[i, :]
        nonlinearity = simulations.data.average_nonlinearity(voltage_curve, current_curve)
        nonlinearities.append(nonlinearity)
    nonlinearities = np.array(nonlinearities)

    for axis, is_high_resistance in zip(axes[:2], [False, True]):
        if is_high_resistance:
            idxs = high_idxs
        else:
            idxs = low_idxs

        V = V_full[idxs, :]
        I = I_full[idxs, :]
        temp_nonlinearities = nonlinearities[idxs]

        for idx in range(V.shape[0]):
            voltage_curve = V[idx, :]
            current_curve = I[idx, :]
            nonlinearity = temp_nonlinearities[idx]
            palette_idx = int(np.floor(N * (nonlinearity - v_min) / (v_max - v_min)))
            axis.plot(
                voltage_curve,
                current_curve,
                linewidth=utils.Config.LINEWIDTH,
                color=palette[palette_idx],
            )

        axis.set_xlim([min_voltage, max_voltage])
        axis.set_ylim(bottom=0)
        axis.set_xlabel(utils.axis_label("voltage"))
        axis.ticklabel_format(axis="y", scilimits=(-1, 1))
        axis.yaxis.get_offset_text().set_fontsize(utils.Config.TICKS_FONT_SIZE)

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=v_min, vmax=v_max))
    cbar = fig.colorbar(sm, ax=axes[:2])
    cbar.set_label(
        label=utils.axis_label("mean-nonlinearity"),
        fontsize=utils.Config.AXIS_LABEL_FONT_SIZE,
    )
    cbar.ax.tick_params(axis="both", which="both", labelsize=utils.Config.TICKS_FONT_SIZE)

    axes[0].set_ylabel(utils.axis_label("current"))
    axes[0].set_title("Low resistance")
    axes[1].set_title("High resistance")

    utils.save_fig(fig, "SiO_x")


def Ta_HfO2():
    fig, axes = utils.fig_init(2, 0.45, fig_shape=(1, 2), sharey=True)

    data = simulations.data.load_Ta_HfO2()
    G_min, G_max = simulations.data.extract_G_off_and_G_on(data)
    vals, p = simulations.data.extract_stuck(data, G_min, G_max)
    logging.info("Stuck probability: %.3g%%", 100 * p)
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
        bbox_to_anchor=(0.5, 1.05),
        handles=handles,
    )

    utils.save_fig(fig, "Ta-HfO2")


def iv_nonlinearity_training(metric="error"):
    fig, axes = utils.fig_init(2, 0.4, fig_shape=(1, 2), sharex=True, sharey=True)

    iterators = simulations.iv_nonlinearity.get_iterators()
    iterators = [iterators[0], iterators[3]]
    inference_idxs = [1, 0]
    titles = ["Standard", "Nonideality-aware"]

    for idx, (iterator, inference_idx, title) in enumerate(zip(iterators, inference_idxs, titles)):
        axis = axes[idx]
        utils.plot_training_curves(axis, iterator, metric=metric, inference_idx=inference_idx)
        axis.set_xlabel(utils.axis_label("epoch"))
        axis.set_title(title)

    axes[0].set_ylabel(utils.axis_label(metric))

    utils.add_legend(
        fig,
        labels=["Training", "Validation", "Test (nonideal)"],
        bbox_to_anchor=(1.1, 0.5),
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

    median_vals = []
    avg_powers = []
    for idx, (iterator, inference_idx) in enumerate(zip(iterators, inference_idxs)):
        avg_power_lst = iterator.test_metric("avg_power", inference_idx=inference_idx)
        y = iterator.test_metric(metric, inference_idx=inference_idx)
        median_vals.append(100 * np.median(y))
        avg_powers.append(np.mean(avg_power_lst))
        color = colors[idx % 3]
        boxplot = utils.plot_boxplot(
            axes, y, color, metric=metric, x=avg_power_lst, is_x_log=True, linear_width=0.2
        )
        boxplots.append(boxplot)

    efficiency = [simulations.utils.get_energy_efficiency(avg_power) for avg_power in avg_powers]
    logging.info("Median %s (%%):", metric)
    logging.info(
        "Low nonlinearity: %.1f (standard), %.1f (aware), %.1f (aware + reg.)", *median_vals[:3]
    )
    logging.info(
        "High nonlinearity: %.1f (standard), %.1f (aware), %.1f (aware + reg.)", *median_vals[3:]
    )
    logging.info("Mean power (W):")
    logging.info(
        "Low nonlinearity: %.3g (standard), %.3g (aware), %.3g (aware + reg.)", *avg_powers[:3]
    )
    logging.info(
        "High nonlinearity: %.3g (standard), %.3g (aware), %.3g (aware + reg.)", *avg_powers[3:]
    )
    logging.info("Efficiency (TOP/(sW)):")
    logging.info(
        "Low nonlinearity: %.3g (standard), %.3g (aware), %.3g (aware + reg.)", *efficiency[:3]
    )
    logging.info(
        "High nonlinearity: %.3g (standard), %.3g (aware), %.3g (aware + reg.)", *efficiency[3:]
    )

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
    median_vals = []
    for idx, (iterator, color) in enumerate(zip(iterators, [colors["vermilion"], colors["blue"]])):
        y = iterator.test_metric(metric)
        median_vals.append(100 * np.median(y))
        _ = utils.plot_boxplot(axis, y, color, x=idx, metric=metric, linewidth_scaling=2 / 3)
    logging.info("Median %s (%%):", metric)
    logging.info("Standard: %.1f", median_vals[0])
    logging.info("Aware: %.1f", median_vals[1])

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Standard", "Nonideality-aware"])

    axis.set_xlabel("Training")

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

        axis.xaxis.set_ticks(np.arange(0.5, 3.0, 0.5))
        axis.yaxis.set_ticks(np.arange(0.5, 3.0, 0.5))

        if idx > 3:
            axis.set_xlabel(utils.axis_label("g-plus", unit_prefix="mu"))
        if idx in [0, 4]:
            axis.set_ylabel(utils.axis_label("g-minus", unit_prefix="mu"))

    mappings = ["standard (avg.)", "standard (min. G)", "double w", "double w (reg.)"]
    variabilities = ["more uniform", "less uniform"]
    for iterator_idxs, axis, variability in zip(
        [[0, 1, 4, 5], [2, 3, 6, 7]], axes[-2:], variabilities
    ):
        logging.info("%s variability:", variability)
        for iterator_idx, color, mapping in zip(iterator_idxs, colors, mappings):
            iterator = iterators[iterator_idx]
            avg_power = iterator.test_metric("avg_power")
            y = iterator.test_metric(metric)
            logging.info("%s mapping: %.1f%% median %s", mapping, 100 * np.median(y), metric)
            utils.plot_boxplot(axis, y, color, x=1000 * avg_power, metric=metric, linear_width=0.2)
            axis.set_xlabel(utils.axis_label("power-consumption", unit_prefix="m"))

    axes[-2].set_ylabel(utils.axis_label(metric))
    axes[-1].sharey(axes[-2])
    axes[-1].sharex(axes[-2])
    axes[-1].label_outer()

    utils.save_fig(fig, "weight-implementation", metric=metric)


def nonideality_agnosticism(metric: str = "error", norm_rows=True, include_val_label=False):
    training_labels = {
        "nonreg__64__none_none__ideal": "Ideal",
        "nonreg__64__0.00069_0.00345__IVNL_PF:-1.12_0.628__-1.34_-25.9": r"Low nonlin.",
        "reg__64__0.00069_0.00345__IVNL_PF:-1.12_0.628__-1.34_-25.9": r"Low nonlin. (reg.)",
        "nonreg__64__5.25e-07_2.62e-06__IVNL_PF:-1.03_-0.251__-0.0901_-37.1": r"High nonlin.",
        "reg__64__5.25e-07_2.62e-06__IVNL_PF:-1.03_-0.251__-0.0901_-37.1": r"High nonlin. (reg.)",
        "nonreg__64__5.25e-07_2.62e-06__StuckOff:0.05": r"Stuck OFF",
        "nonreg__64__4.36e-05_0.000978__StuckDistr:0.101_1.77e-05": r"Stuck [experimental]",
        "nonreg__64__5.25e-07_2.62e-06__D2DLN:0.25_0.25": "More uniform D2D var.",
        "reg__64__5.25e-07_2.62e-06__D2DLN:0.25_0.25": "More uniform D2D var. (reg.)",
        "nonreg__64__5.25e-07_2.62e-06__D2DLN:0.05_0.5": "Less uniform D2D var.",
        "reg__64__5.25e-07_2.62e-06__D2DLN:0.05_0.5": "Less uniform D2D var. (reg.)",
        "nonreg__64__5.25e-07_2.62e-06__IVNL_PF:-1.03_-0.251__-0.0901_-37.1+StuckOn:0.05": r"High nonlin. + stuck ON",
        "nonreg__64__5.25e-07_2.62e-06__D2DLN:0.5_0.5": "High D2D var.",
    }
    inference_labels = {
        "none_none__ideal": training_labels["nonreg__64__none_none__ideal"],
        "0.00069_0.00345__IVNL_PF:-1.12_0.628__-1.34_-25.9": training_labels[
            "nonreg__64__0.00069_0.00345__IVNL_PF:-1.12_0.628__-1.34_-25.9"
        ],
        "5.25e-07_2.62e-06__IVNL_PF:-1.03_-0.251__-0.0901_-37.1": training_labels[
            "nonreg__64__5.25e-07_2.62e-06__IVNL_PF:-1.03_-0.251__-0.0901_-37.1"
        ],
        "5.25e-07_2.62e-06__StuckOff:0.05": training_labels[
            "nonreg__64__5.25e-07_2.62e-06__StuckOff:0.05"
        ],
        "4.36e-05_0.000978__StuckDistr:0.101_1.77e-05": training_labels[
            "nonreg__64__4.36e-05_0.000978__StuckDistr:0.101_1.77e-05"
        ],
        "5.25e-07_2.62e-06__D2DLN:0.25_0.25": training_labels[
            "nonreg__64__5.25e-07_2.62e-06__D2DLN:0.25_0.25"
        ],
        "5.25e-07_2.62e-06__D2DLN:0.05_0.5": training_labels[
            "nonreg__64__5.25e-07_2.62e-06__D2DLN:0.05_0.5"
        ],
        "5.25e-07_2.62e-06__IVNL_PF:-1.03_-0.251__-0.0901_-37.1+StuckOn:0.05": training_labels[
            "nonreg__64__5.25e-07_2.62e-06__IVNL_PF:-1.03_-0.251__-0.0901_-37.1+StuckOn:0.05"
        ],
        "5.25e-07_2.62e-06__D2DLN:0.5_0.5": training_labels[
            "nonreg__64__5.25e-07_2.62e-06__D2DLN:0.5_0.5"
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
    axes.set_title(utils.axis_label("inference-error"), y=-0.11)

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


def pf_residuals(is_d_times_perm: bool = False):
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])

    gs_top = gs[0].subgridspec(2, 2)
    gs_bottom = gs[1].subgridspec(2, 2)

    subplots = list(gs_top) + list(gs_bottom)
    for subplot in subplots:
        fig.add_subplot(subplot)

    fig, axes = utils.fig_init(2, 1.0, custom_fig=fig)

    axes[1].sharey(axes[0])
    plt.setp(axes[1].get_yticklabels(), visible=False)
    axes[3].sharey(axes[2])
    plt.setp(axes[3].get_yticklabels(), visible=False)

    axes[2].sharex(axes[0])
    plt.setp(axes[0].get_xticklabels(), visible=False)
    axes[3].sharex(axes[1])
    plt.setp(axes[1].get_xticklabels(), visible=False)

    axes[6].sharex(axes[4])
    plt.setp(axes[4].get_xticklabels(), visible=False)
    axes[6].set_xlim([-2, 2])
    axes[7].sharex(axes[5])
    plt.setp(axes[5].get_xticklabels(), visible=False)
    axes[7].set_xlim([-2, 2])

    exp_data = simulations.data.load_SiO_x_multistate()
    V, I = simulations.data.all_SiO_x_curves(exp_data, clean_data=True)
    resistances, c, d_times_perm, _, _ = simulations.data.pf_relationship(V, I)
    R_0 = const.physical_constants["inverse of conductance quantum"][0]
    # Separate data into before and after the conductance quantum.
    sep_idx = np.searchsorted(resistances, R_0)

    colors = utils.color_dict()

    axes[0].set_ylabel(utils.axis_label("ln-c-residuals"))
    axes[2].set_ylabel(utils.axis_label("ln-d-times-perm-residuals"))
    axes[2].set_xlabel(utils.axis_label("ln-R"))
    axes[3].set_xlabel(utils.axis_label("ln-R"))
    axes[6].set_xlabel(utils.axis_label("theoretical-normal-quartiles"))
    axes[4].set_ylabel(utils.axis_label("ordered-ln-c-residuals"))
    axes[7].set_xlabel(utils.axis_label("theoretical-normal-quartiles"))
    axes[6].set_ylabel(utils.axis_label("ordered-ln-d-times-perm-residuals"))

    for is_high_resistance, ax_idx in zip([False, True], [0, 1]):
        _, _, slopes, intercepts, _ = simulations.data.pf_params(
            exp_data, is_high_resistance, simulations.data.SiO_x_G_on_G_off_ratio()
        )

        if is_high_resistance:
            idxs = np.arange(sep_idx, len(resistances))
            color = colors["reddish-purple"]
        else:
            idxs = np.arange(sep_idx)
            color = colors["blue"]

        x = np.log(resistances[idxs])
        c_points = np.log(c[idxs])
        c_residuals = c_points - slopes[0] * x - intercepts[0]

        d_times_perm_points = np.log(d_times_perm[idxs])
        d_times_perm_residuals = d_times_perm_points - slopes[1] * x - intercepts[1]

        utils.plot_scatter(axes[ax_idx], x, c_residuals, color, scale=10)
        zero_line = [0] * len(x)
        axes[ax_idx].plot(
            x,
            zero_line,
            linewidth=utils.Config.LINEWIDTH,
            color=color,
            linestyle="dashed",
        )

        utils.plot_scatter(axes[2 + ax_idx], x, d_times_perm_residuals, color, scale=10)
        axes[2 + ax_idx].plot(
            x,
            zero_line,
            linewidth=utils.Config.LINEWIDTH,
            color=color,
            linestyle="dashed",
        )

        (osm, osr), (m, b, _) = stats.probplot(c_residuals, dist="norm", plot=None)
        utils.plot_scatter(axes[4 + ax_idx], osm, osr, color, scale=10)
        axes[4 + ax_idx].plot(
            osm,
            m * osm + b,
            linewidth=utils.Config.LINEWIDTH,
            color=color,
            linestyle="dashed",
        )

        (osm, osr), (m, b, _) = stats.probplot(d_times_perm_residuals, dist="norm", plot=None)
        utils.plot_scatter(axes[6 + ax_idx], osm, osr, color, scale=10)
        axes[6 + ax_idx].plot(
            osm,
            m * osm + b,
            linewidth=utils.Config.LINEWIDTH,
            color=color,
            linestyle="dashed",
        )

    for ax in [axes[0], axes[2], axes[4], axes[5], axes[6], axes[7]]:
        low, high = ax.get_ylim()
        bound = 1.15 * max(abs(low), abs(high))
        ax.set_ylim(-bound, bound)

    utils.save_fig(fig, "pf-residuals")
