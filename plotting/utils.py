import matplotlib.transforms as mtransforms


def color_list():
    """Okabe-Ito colorblind-friendly palette.

    Returns
    ----------
    list of string
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


def color_dict():
    """Same as `colors_list()` but dict.

    Returns
    ----------
    dict of string
        HEX color codes.
    """
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