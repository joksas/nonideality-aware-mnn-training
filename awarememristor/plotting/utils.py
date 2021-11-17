import os
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


def _cm_to_in(length: float) -> float:
    return length / 2.54


class Config:
    AXIS_LABEL_FONT_SIZE: float = 12
    LEGEND_FONT_SIZE: float = 8
    TICKS_FONT_SIZE: float = 8
    SUBPLOT_LABEL_SIZE: float = 12
    LINEWIDTH: float = 0.75
    MARKER_SIZE: float = 0.5
    BOXPLOT_LINEWIDTH: float = 0.75
    # Advanced Science
    COL_WIDTHS: dict[int, float] = {
        1: _cm_to_in(8.5),
        2: _cm_to_in(17.8),
    }
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


def fig_init(
    width_num_cols: int,
    height_frac: float,
    fig_shape: tuple[int, int] = (1, 1),
    sharex=False,
    sharey=False,
) -> tuple[matplotlib.figure, matplotlib.axes]:
    width = Config.COL_WIDTHS[width_num_cols]
    height = height_frac * width

    fig, axes = plt.subplots(
        *fig_shape,
        sharex=sharex,
        sharey=sharey,
        figsize=(width, height),
    )
    fig.tight_layout()

    if fig_shape == (1, 1):
        temp_axes = np.array([axes])
    else:
        temp_axes = axes

    for idx, axis in enumerate(temp_axes.flatten()):
        axis.xaxis.label.set_size(Config.AXIS_LABEL_FONT_SIZE)
        axis.yaxis.label.set_size(Config.AXIS_LABEL_FONT_SIZE)
        axis.tick_params(axis="both", which="both", labelsize=Config.TICKS_FONT_SIZE)
        if fig_shape != (1, 1):
            add_subfigure_label(fig, axis, idx, Config.SUBPLOT_LABEL_SIZE)

    return fig, axes


def add_subfigure_label(
    fig,
    axis,
    letter_idx: int,
    fontsize: float = Config.SUBPLOT_LABEL_SIZE,
    is_lowercase: bool = True,
):
    trans = mtransforms.ScaledTranslation(-16 / 72, 2 / 72, fig.dpi_scale_trans)
    ascii_idx = 65 + letter_idx
    if is_lowercase:
        ascii_idx += 32
    axis.text(
        0.0,
        1.0,
        chr(ascii_idx),
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

    # Network might have been trained with a different number of callbacks.
    nonideality_label = iterator.inferences[inference_idx].nonideality_label()
    for idx, history in enumerate(iterator.info()["callback_infos"]["memristive_test"]["history"]):
        if history["nonideality_label"] == nonideality_label:
            true_inference_idx = idx
            break

    x_training_testing, y_training_testing = iterator.training_testing_curves(
        metric, true_inference_idx
    )
    plot_curve(
        axis, x_training_testing, y_training_testing, colors["reddish-purple"], metric=metric
    )

    axis.set_yscale("log")
    axis.set_xlim([0, len(x_training)])


def plot_curve(axis, x, y, color, metric="error"):
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


def numpify(x):
    try:  # In case `tf.Tensor`
        x = x.numpy()
    except AttributeError:
        pass

    return x


def plot_scatter(axis, x, y, color):
    x = numpify(x)
    y = numpify(y)
    x = x.flatten()
    y = y.flatten()
    axis.scatter(
        x, y, color=color, marker="x", s=Config.MARKER_SIZE, linewidth=Config.MARKER_SIZE / 2
    )


def plot_boxplot(axis, y, color, x=None, metric="error", is_x_log=False, linewidth_scaling=1.0):
    y = y.flatten()
    if metric in ["accuracy", "error"]:
        y = 100 * y

    linear_width = 0.2
    positions = None
    if x is not None:
        try:
            x = x.flatten()
        except AttributeError:
            pass
        positions = [np.mean(x)]
    widths = None
    if is_x_log and positions is not None:
        widths = [
            10 ** (np.log10(positions[0]) + linear_width / 2.0)
            - 10 ** (np.log10(positions[0]) - linear_width / 2.0)
        ]
    boxplot = axis.boxplot(
        y,
        positions=positions,
        widths=widths,
        sym=color,
    )
    plt.setp(boxplot["fliers"], marker="x", markersize=1, markeredgewidth=0.5)
    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(
            boxplot[element], color=color, linewidth=linewidth_scaling * Config.BOXPLOT_LINEWIDTH
        )

    if is_x_log:
        axis.set_xscale("log")
    axis.set_yscale("log")

    return boxplot


def add_boxplot_legend(axis, boxplots, labels, linewdith=1.0, loc="upper right"):
    leg = axis.legend(
        [boxplot["boxes"][0] for boxplot in boxplots],
        labels,
        fontsize=Config.LEGEND_FONT_SIZE,
        frameon=False,
        loc=loc,
    )
    for line in leg.get_lines():
        line.set_linewidth(linewdith)


def add_legend(
    fig,
    labels=None,
    ncol=1,
    loc="center",
    bbox_to_anchor=(0.5, 1.0),
    linewidth=1.0,
    frameon=False,
    handles=None,
):
    if handles is None:
        leg = fig.legend(labels, ncol=ncol, loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=frameon)
    else:
        leg = fig.legend(
            handles=handles, ncol=ncol, loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=frameon
        )
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
    elif var_name == "inference":
        label = "inference"
    elif var_name == "training":
        label = "training"
    elif var_name == "power-consumption":
        label = "ohmic power consumption (W)"
    elif var_name == "d2d-uniformity":
        label = "uniformity of D2D variability"
    elif var_name == "checkpoint":
        label = "checkpoint"
    elif var_name == "conductance":
        label = "conductance (mS)"
    elif var_name == "voltage":
        label = "voltage (V)"
    elif var_name == "current":
        label = "current (A)"
    elif var_name == "count":
        label = "count (#)"
    elif var_name == "nonlinearity-parameter":
        label = "nonlinearity parameter"
    elif var_name == "pulse-number":
        label = "pulse number (#)"
    else:
        raise ValueError(f'Unrecognised variable name "{var_name}".')

    if prepend is not None:
        label = f"{prepend} {label}"

    first_letter = label[0].upper()
    if len(label) > 1:
        label = first_letter + label[1:]
    else:
        label = first_letter

    return label


def annotate_heatmap(
    im: matplotlib.image,
    data: np.array = None,
    valfmt: Union[str, matplotlib.ticker.StrMethodFormatter] = "{x:.2f}",
    textcolors: tuple[str, str] = ("black", "white"),
    threshold: float = None,
    **textkw,
):
    """Annotate a heatmap. Adapted from
    <https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html>.

    Args:
        im: The image to be labelled.
        data: Data used to annotate. If `None`, the image's data is used.
        valfmt: The format of the annotations inside the heatmap. This should
            either use the string format method, e.g. `{x:.2f}`, or be a
            `matplotlib.ticker.Formatter`.
        textcolors: A pair of colors. The first is used for values below a
            threshold, the second for those above.
        threshold: Value in data units according to which the colors from
            textcolors are applied. If `None` (the default) uses the middle of
            the colormap as separation.
        **kwargs: All other arguments are forwarded to each call to `text` used to create the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def add_heatmap(fig, axis, data, x_ticks=None, y_ticks=None, metric="error"):
    if metric in ["accuracy", "error"]:
        data = 100 * data

    image = axis.imshow(data, norm=matplotlib.colors.LogNorm(), cmap="cividis")

    if x_ticks is not None:
        axis.set_xticks(np.arange(len(x_ticks)))
        axis.set_xticklabels(x_ticks)
        axis.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(axis.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
        axis.xaxis.set_label_position("top")
    if y_ticks is not None:
        axis.set_yticks(np.arange(len(y_ticks)))
        axis.set_yticklabels(y_ticks)

    cbar = fig.colorbar(image, ax=axis)
    cbar.ax.set_ylabel(
        axis_label(metric, prepend="median"),
        rotation=-90,
        fontsize=Config.AXIS_LABEL_FONT_SIZE,
        va="bottom",
    )
    cbar.ax.tick_params(axis="both", which="both", labelsize=Config.TICKS_FONT_SIZE)

    annotate_heatmap(
        image, valfmt="{x:.1f}", textcolors=("white", "black"), size=Config.TICKS_FONT_SIZE
    )


def add_histogram(axis, values: np.ndarray, color: str, bins: int = 100, alpha: float = 1.0):
    try:  # In case `tf.Tensor`
        values = values.numpy()
    except AttributeError:
        pass
    values = values.flatten()
    axis.hist(values, bins=bins, color=color, alpha=alpha)