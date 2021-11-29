import copy
import os
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from numpy import ma


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
    ONE_COLUMN_WIDTH: float = _cm_to_in(8.5)
    TWO_COLUMNS_WIDTH: float = _cm_to_in(17.8)
    TWO_THIRDS_COLUMN_WIDTH: float = 2 / 3 * TWO_COLUMNS_WIDTH
    COL_WIDTHS: dict[Union[int, tuple[int, int]], float] = {
        1: ONE_COLUMN_WIDTH,
        2: TWO_COLUMNS_WIDTH,
        (2, 3): TWO_THIRDS_COLUMN_WIDTH,
    }


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


def get_linestyles():
    return [
        "solid",
        "dotted",
        "dashed",
        "dashdot",
        (0, (1, 10)),  # "loosely dotted"
    ]


def fig_init(
    width_num_cols: Union[int, tuple[int, int]],
    height_frac: float,
    fig_shape: tuple[int, int] = (1, 1),
    sharex=False,
    sharey=False,
    scaled_translation: tuple[float, float] = (-12 / 72, 2 / 72),
    custom_fig: matplotlib.figure = None,
) -> tuple[matplotlib.figure, matplotlib.axes]:
    width = Config.COL_WIDTHS[width_num_cols]
    height = height_frac * width
    if custom_fig is not None:
        fig = custom_fig
        fig.set_size_inches(width, height)
        axes = fig.axes
        axes = np.array(axes)
        fig_shape = axes.shape
    else:
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
            add_subfigure_label(fig, axis, idx, scaled_translation, Config.SUBPLOT_LABEL_SIZE)

    return fig, axes


def add_subfigure_label(
    fig,
    axis,
    letter_idx: int,
    scaled_translation: tuple[float, float],
    fontsize: float = Config.SUBPLOT_LABEL_SIZE,
    is_lowercase: bool = True,
):
    trans = mtransforms.ScaledTranslation(*scaled_translation, fig.dpi_scale_trans)
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


def plot_training_curves(
    fig,
    axis,
    iterator,
    subfigure_idx=None,
    metric="error",
    inference_idx=0,
    linestyle="solid",
    is_many=False,
):
    colors = color_dict()

    # Training curve.
    x_training, y_training = iterator.training_curves(metric)
    plot_curve(
        axis,
        x_training,
        y_training,
        colors["orange"],
        metric=metric,
        linestyle=linestyle,
        is_many=is_many,
    )

    # Validation curve.
    x_validation, y_validation = iterator.validation_curves(metric)
    plot_curve(
        axis,
        x_validation,
        y_validation,
        colors["sky-blue"],
        metric=metric,
        linestyle=linestyle,
        is_many=is_many,
    )

    # Testing (during training) curve.
    x_training_testing, y_training_testing = iterator.training_testing_curves(
        metric, iterator.inferences[inference_idx]
    )
    plot_curve(
        axis,
        x_training_testing,
        y_training_testing,
        colors["reddish-purple"],
        metric=metric,
        linestyle=linestyle,
        is_many=is_many,
    )

    axis.set_yscale("log")
    axis.set_xlim([0, len(x_training)])


def plot_curve(axis, x, y, color, metric="error", linestyle="solid", is_many=False):
    if metric in ["accuracy", "error"]:
        y = 100 * y
    lw = Config.LINEWIDTH
    if is_many:
        lw /= 2
    if len(y.shape) > 1:
        alpha = 0.25
        if is_many:
            alpha /= 2
        y_min = np.min(y, axis=1)
        y_max = np.max(y, axis=1)
        y_median = np.median(y, axis=1)
        axis.fill_between(x, y_min, y_max, color=color, alpha=alpha, linewidth=0)
        axis.plot(x, y_median, color=color, linewidth=lw / 2, linestyle=linestyle)
    else:
        if is_many:
            x = np.concatenate((x[::20], [x[-1]]))
            y = np.concatenate((y[::20], [y[-1]]))
            lw /= 2  # Make all curves the same linewidth.
        axis.plot(x, y, color=color, linewidth=lw, linestyle=linestyle)


def numpify(x):
    try:  # In case `tf.Tensor`
        x = x.numpy()
    except AttributeError:
        pass

    return x


def plot_scatter(axis, x, y, color, alpha=1.0, random_proportion=None):
    x = numpify(x)
    y = numpify(y)
    x = x.flatten()
    y = y.flatten()
    if random_proportion:
        np.random.seed(0)
        num_points = x.size
        num_reduced_points = int(random_proportion * num_points)
        random_idxs = np.random.choice(num_points, num_reduced_points)
        x = x[random_idxs]
        y = y[random_idxs]
    axis.scatter(
        x,
        y,
        color=color,
        marker="x",
        s=Config.MARKER_SIZE,
        linewidth=Config.MARKER_SIZE,
        alpha=alpha,
    )


def plot_boxplot(
    axis,
    y,
    color,
    x=None,
    metric="error",
    is_x_log=False,
    linewidth_scaling=1.0,
    linear_width: float = 0.25,
):
    y = y.flatten()
    if metric in ["accuracy", "error"]:
        y = 100 * y

    positions = None
    if x is not None:
        try:
            x = x.flatten()
        except AttributeError:
            pass
        positions = [np.mean(x)]
    if is_x_log and positions is not None:
        widths = [
            10 ** (np.log10(positions[0]) + linear_width / 2.0)
            - 10 ** (np.log10(positions[0]) - linear_width / 2.0)
        ]
    else:
        widths = [linear_width]
    boxplot = axis.boxplot(
        y,
        positions=positions,
        widths=widths,
        sym=color,
    )
    plt.setp(boxplot["fliers"], marker="x", markersize=4 * Config.MARKER_SIZE, markeredgewidth=0.5)
    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(
            boxplot[element], color=color, linewidth=linewidth_scaling * Config.BOXPLOT_LINEWIDTH
        )

    if is_x_log:
        axis.set_xscale("log")
    else:
        axis.set_xscale("linear")
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


def save_fig(fig, name: str, is_supporting: bool = False, metric: str = "error"):
    dir_name = "plots"
    os.makedirs(dir_name, exist_ok=True)
    if metric != "error":
        name += f"-{metric}"
    if is_supporting:
        name = f"supporting-information--{name}"
    path = os.path.join(dir_name, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight", transparent=True)


def axis_label(var_name: str, prepend: str = None, unit_prefix: str = "") -> str:
    if var_name == "accuracy":
        label = "accuracy (%)"
    elif var_name == "error":
        label = "error (%)"
    elif var_name == "loss":
        label = "loss"
    elif var_name == "epoch":
        label = "epoch"
    elif var_name == "inference":
        label = "inference"
    elif var_name == "training":
        label = "training"
    elif var_name == "power-consumption":
        label = f"power consumption ({unit_prefix}W)"
    elif var_name == "d2d-uniformity":
        label = "uniformity of D2D variability"
    elif var_name == "checkpoint":
        label = "checkpoint"
    elif var_name == "conductance":
        label = f"conductance ({unit_prefix}S)"
    elif var_name == "voltage":
        label = f"voltage ({unit_prefix}V)"
    elif var_name == "current":
        label = f"current ({unit_prefix}A)"
    elif var_name == "nonlinearity-parameter":
        label = "nonlinearity parameter"
    elif var_name == "pulse-number":
        label = "pulse number"
    elif var_name == "g-plus":
        label = rf"$G_{{+}}$ ({unit_prefix}S)"
    elif var_name == "g-minus":
        label = rf"$G_{{-}}$ ({unit_prefix}S)"
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


def get_luminance(r, g, b):
    """Adapted from <https://stackoverflow.com/a/596243/17322548>."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def annotate_heatmap(
    im: matplotlib.image,
    data: np.array = None,
    valfmt: Union[str, matplotlib.ticker.StrMethodFormatter] = "{x:.2f}",
    textcolors: tuple[str, str] = ("black", "white"),
    threshold: float = 0.5,
    norm_rows: bool = False,
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
        if norm_rows:
            colors = im.cmap(matplotlib.colors.LogNorm()(data[i, :]))
            luminance = get_luminance(colors[:, 0], colors[:, 1], colors[:, 2])
        else:
            colors = im.cmap(matplotlib.colors.LogNorm()(data))
            luminance = get_luminance(colors[:, :, 0], colors[:, :, 1], colors[:, :, 2])
        for j in range(data.shape[1]):
            if norm_rows:
                cell_luminance = luminance[j]
            else:
                cell_luminance = luminance[i, j]
            kw.update(color=textcolors[int(cell_luminance > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def add_heatmap(fig, axis, data, x_ticks=None, y_ticks=None, metric="error", norm_rows=False):
    if metric in ["accuracy", "error"]:
        data = 100 * data

    data = data.to_numpy()
    if norm_rows:
        num_rows = data.shape[0]
        row_indices = np.arange(num_rows)
        for i in range(num_rows):
            row_data = ma.array(copy.deepcopy(data))
            row_data[row_indices != i, :] = ma.masked
            image = axis.imshow(row_data, norm=matplotlib.colors.LogNorm(), cmap="cividis")
    else:
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

    if not norm_rows:
        cbar = fig.colorbar(image, ax=axis)
        cbar.ax.set_ylabel(
            axis_label(metric, prepend="median"),
            rotation=-90,
            fontsize=Config.AXIS_LABEL_FONT_SIZE,
            va="bottom",
        )
        cbar.ax.tick_params(axis="both", which="both", labelsize=Config.TICKS_FONT_SIZE)

    annotate_heatmap(
        image,
        data=data,
        valfmt="{x:.1f}",
        textcolors=("white", "black"),
        size=Config.TICKS_FONT_SIZE,
        norm_rows=norm_rows,
    )


def add_histogram(axis, values: np.ndarray, color: str, bins: int = 100, alpha: float = 1.0):
    try:  # In case `tf.Tensor`
        values = values.numpy()
    except AttributeError:
        pass
    values = values.flatten()
    axis.hist(values, bins=bins, color=color, alpha=alpha)
