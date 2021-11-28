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
