def color_list():
    """Okabe-Ito colorblind-friendly palette.

    Returns
    ----------
    list of string
        HEX color codes.
    """
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    return colors


def color_dict():
    """Same as `colors_list()` but dict.

    Returns
    ----------
    dict of string
        HEX color codes.
    """
    color_names = ["orange", "sky-blue", "bluish-green", "yellow", "blue", "vermilion", "reddish-purple", "black"]
    colors = dict(zip(color_names, color_list()))
    return colors