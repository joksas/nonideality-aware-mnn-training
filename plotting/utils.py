import os
from pathlib import Path
from typing import Union

import matplotlib.transforms as mtransforms
import numpy as np


def _cm_to_in(length) -> float:
    return length / 2.54


class Config:
    AXIS_LABEL_FONT_SIZE: float = 12
    LEGEND_FONT_SIZE: float = 8
    TICKS_FONT_SIZE: float = 8
    SUBPLOT_LABEL_SIZE: float = 12
    LINEWIDTH: float = 0.75
    # Advanced Science
    ONE_COLUMN_WIDTH: float = _cm_to_in(8.5)
    TWO_COLUMNS_WIDTH: float = _cm_to_in(17.8)


def color_list() -> list[str]:
    """Return colors of Okabe-Ito colorblind-friendly palette.

    Returns:
        HEX color codes.
    """
    colors = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]
    return colors


def color_dict() -> dict[str, str]:
    """Return same as `colors_list()` but dict."""
    color_names = [
        "orange",
        "sky-blue",
        "bluish-green",
        "yellow",
        "blue",
        "vermilion",
        "reddish-purple",
        "black",
    ]
    colors = dict(zip(color_names, color_list()))
    return colors


def add_subfigure_label(fig, axis, letter_idx, fontsize):
    trans = mtransforms.ScaledTranslation(-16 / 72, 2 / 72, fig.dpi_scale_trans)
    axis.text(
        0.0,
        1.0,
        chr(letter_idx + 65),
        transform=axis.transAxes + trans,
        fontweight="bold",
        fontsize=fontsize,
    )


def plot_training_curves(fig, axis, iterator, subfigure_idx=None, metric="error", inference_idx=0):
    colors = color_dict()

    # Training curve.
    x_training, y_training = iterator.training_curves(metric)
    plot_curve(axis, x_training, y_training, colors["orange"], metric=metric)

    # Validation curve.
    x_validation, y_validation = iterator.validation_curves(metric)
    plot_curve(axis, x_validation, y_validation, colors["sky-blue"], metric=metric)

    # Testing (during training) curve.
    x_training_testing, y_training_testing = iterator.training_testing_curves(metric, inference_idx)
    plot_curve(
        axis, x_training_testing, y_training_testing, colors["reddish-purple"], metric=metric
    )

    axis.set_yscale("log")
    axis.tick_params(axis="both", which="both", labelsize=Config.TICKS_FONT_SIZE)
    axis.set_xlim([0, len(x_training)])

    if subfigure_idx is not None:
        add_subfigure_label(fig, axis, subfigure_idx, Config.SUBPLOT_LABEL_SIZE)


def plot_curve(axis, x, y, color, metric=None):
    if metric in ["accuracy", "error"]:
        y = 100 * y
    if len(y.shape) > 1:
        y_min = np.min(y, axis=1)
        y_max = np.max(y, axis=1)
        y_median = np.median(y, axis=1)
        axis.fill_between(x, y_min, y_max, color=color, alpha=0.25, linewidth=0)
        axis.plot(x, y_median, color=color, linewidth=Config.LINEWIDTH / 2)
    else:
        axis.plot(x, y, color=color, linewidth=Config.LINEWIDTH)


def add_legend(
    fig, labels, ncol=1, loc="center", bbox_to_anchor=(0.5, 1.0), linewidth=1.0, frameon=False
):
    leg = fig.legend(labels, ncol=ncol, loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=frameon)
    for line in leg.get_lines():
        line.set_linewidth(linewidth)


def save_fig(fig, name: str):
    path = os.path.join(Path(__file__).parent, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight", transparent=True)


def axis_label(var_name: str, prepend: str = None) -> str:
    if var_name == "accuracy":
        label = "accuracy (%)"
    elif var_name == "error":
        label = "error (%)"
    elif var_name == "loss":
        label = "loss"
    elif var_name == "epoch":
        label = "epoch (#)"
    elif var_name == "training":
        label = "training"
    else:
        raise ValueError(f'Unrecognised variable name "{var_name}".')

    if prepend is not None:
        label = f"{prepend} {label}"

    return label.capitalize()