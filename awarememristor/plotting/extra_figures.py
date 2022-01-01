from matplotlib.lines import Line2D

from awarememristor import simulations
from awarememristor.plotting import utils


def SiO_x_nonlinearity_dependence():
    fig, axes = utils.fig_init(1, 0.8)

    data = simulations.data.load_SiO_x_multistate()

    min_voltage, max_voltage = 0.0, 0.5

    curves = simulations.data.low_high_n_SiO_x_curves(data)
    colors = [utils.color_dict()[key] for key in ["blue", "vermilion"]]

    for (voltages, currents), color in zip(curves, colors):
        for idx in range(voltages.shape[0]):
            voltage_curve = voltages[idx, :]
            current_curve = currents[idx, :]
            ref_voltages = voltage_curve[2::2]
            num_points = len(voltage_curve)
            nl_pars = current_curve[2::2] / current_curve[1 : int(num_points / 2) + 1]
            axes.plot(
                ref_voltages,
                nl_pars,
                linewidth=utils.Config.LINEWIDTH,
                color=color,
            )

    axes.set_xlim([min_voltage, max_voltage])
    axes.set_ylim(bottom=0)
    axes.set_xlabel(utils.axis_label("voltage", prepend="reference"))
    axes.ticklabel_format(axis="y", scilimits=(-1, 1))
    axes.yaxis.get_offset_text().set_fontsize(utils.Config.TICKS_FONT_SIZE)

    handles = [
        Line2D([0], [0], color=colors[0], label="Low nonlinearity"),
        Line2D([0], [0], color=colors[1], label="High nonlinearity"),
    ]

    utils.add_legend(
        fig,
        ncol=2,
        bbox_to_anchor=(0.5, 1.05),
        handles=handles,
    )

    axes.set_ylabel(utils.axis_label("nonlinearity-parameter"))

    utils.save_fig(fig, "extra--SiO_x-nonlinearity-dependence")
