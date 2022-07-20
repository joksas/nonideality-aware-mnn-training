import logging

import numpy as np
from matplotlib.pyplot import rc

from awarememristor import crossbar, simulations
from awarememristor.plotting import utils
from awarememristor.training import architecture

logging.getLogger().setLevel(logging.INFO)

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})


def weight_implementation(metric="error"):
    fig, axes = utils.fig_init(2, 0.35, fig_shape=(1, 3), sharex=True, sharey=True)

    for axis in axes[:8]:
        axis.label_outer()
        axis.set_aspect("equal", adjustable="box")

    iterators = simulations.weight_implementation.get_iterators()[1:]
    iterators = [iterators[idx] for idx in [0, 6, 5]]
    colors = [utils.color_dict()[key] for key in ["orange", "blue", "bluish-green"]]
    titles = [
        "Conventional weights",
        "Double weights\nadapting to noise\nat low conductance",
        "Double weights\nwith regularisation",
    ]

    for idx, (axis, iterator, color, title) in enumerate(zip(axes, iterators, colors, titles)):
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

        axis.set_xlabel(utils.axis_label("g-plus", unit_prefix="mu"))
        if idx == 0:
            axis.set_ylabel(utils.axis_label("g-minus", unit_prefix="mu"))

        axis.set_title(title)

    utils.save_fig(fig, "extra--weight-implementation", metric=metric)
