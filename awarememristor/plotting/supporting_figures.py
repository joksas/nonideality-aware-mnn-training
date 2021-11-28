import matplotlib.pyplot as plt
import numpy as np

from awarememristor import simulations
from awarememristor.plotting import utils


def iv_nonlinearity_training_curves(metric="error"):
    fig, axes = utils.fig_init(2, 0.55, fig_shape=(2, 3), sharex=True, sharey=True)

    iterators = simulations.iv_nonlinearity.get_iterators()
    # Same training, different inference.
    iterators.insert(3, iterators[0])
    inference_idxs = [0, 0, 0, 1, 0, 0]

    for training_idx, linestyle in enumerate(utils.get_linestyles()):
        for i in range(len(iterators)):
            iterators[i].training.repeat_idx = training_idx

        for idx, (iterator, inference_idx) in enumerate(zip(iterators, inference_idxs)):
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

    utils.save_fig(fig, f"iv-nonlinearity-training-{metric}", is_supporting=True)


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
