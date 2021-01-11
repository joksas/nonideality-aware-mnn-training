import numpy as np


def extract(
        type_='equal_G', ratio=10, num_states=100, percentile=0.99, list_=None):
    """Extracts discrete weights levels.

    Parameters
    ----------
    type_ : {'equal_G', 'equal_R', 'custom'}, optional
        Distribution of conductances levels.
    ratio : float, optional
        Ratio between the highest and lowest resistance (or conductance)
        states.
    num_states : int or np.inf, optional
        Number of discrete conductance states.
    percentile : float, optional
        Percentile of sorted absolute weights (normalized to 1).
    list_ : list of float
        If type_ = 'custom', then this parameter is used to create the levels.

    Returns
    -------
    named tuple
        Contains fields list_, ratio, num_states and percentile.
    """
    if num_states != np.inf:
        if type_ == 'equal_G':
            positive_list = equal_G(ratio, num_states)
        elif type_ == 'equal_R':
            positive_list = equal_R(ratio, num_states)
        elif type_ == 'custom':
            positive_list, ratio, num_states = custom(list_)

        negative_list = -np.flip(positive_list)
        list_ = np.concatenate((negative_list, [0], positive_list), axis=None)
    return list_


def equal_G(ratio, num_states):
    """Extracts equally spaced conductance states.

    Parameters
    ----------
    ratio : float
        Ratio between the highest and lowest resistance (or conductance)
        states.
    num_states : int
        Number of discrete conductance states.

    Returns
    -------
    ndarray
        List of states (normalized to 1).
    """
    upper = 1
    if num_states != 1:
        lower = 1 / ratio
    else:
        lower = upper
    return np.linspace(lower, upper, num_states)


def equal_R(ratio, num_states):
    """Extracts equally spaced resistance states.

    Parameters
    ----------
    ratio : float
        Ratio between the highest and lowest resistance (or conductance)
        states.
    num_states : int
        Number of discrete conductance states.

    Returns
    -------
    ndarray
        List of states (normalized to 1).
    """
    upper = ratio
    lower = 1
    return np.flip(1./np.linspace(lower, upper, num_states))


def custom(list_=None):
    """Extracts states with custom spacing.

    Parameters
    ----------
    list_ : list of float, optional
        If not None, then this parameter is used to create the levels.

    Returns
    -------
    ndarray
        List of states (normalized to 1).
    """
    if list_ is None:
        raise ValueError("If `distr == \"custom\"` and `num_states != np.inf ,"\
                "then `list_` cannot be `None`!")
    else:
        positive_list = np.array(list_)
        positive_list = positive_list/positive_list[-1]
        ratio = positive_list[-1]/positive_list[0]
        num_states = positive_list.size

    return positive_list, ratio, num_states

