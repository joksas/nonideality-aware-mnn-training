import numpy as np

from badmemristor import levels


def w_to_G(weights, max_weight, G_min, G_max, distr="equal_G", num_states=np.inf, list_=None):
    """Maps weights onto conductances.

    Parameters
    ----------
    weights : ndarray
        Synaptic weights of a single synaptic layer of size `m x n`.
    max_weight : float
        Assumed maximum weight.
    G_min : float
        Minimum conductance.
    G_max : float
        Maximum conductance.
    distr : {'equal_G', 'equal_R', 'custom'}, optional
        The spacing between states. 'equal_G' denotes equal spacing between
        conductance states, 'equal_R' denotes equal spacing between resistance
        states, and 'custom' denotes custom spacing between conductance states
        defined by argument `list_`.
    num_states : float or int, optional
        Number of states (including G_min and G_max).
    list_ : list of float, optional
        Relative magnitudes of conductance levels if `distr == 'equal_G'`.

    Returns
    ----------
    ndarray
        Conductances in shape `m x 2n`.
    """
    HRS_LRS = G_max/G_min
    weights = discretize_weights(weights, max_weight, distr, HRS_LRS, num_states, list_)
    scaling_factor = max_weight/G_max
    scaled_weights = weights/scaling_factor
    G = scaled_w_to_G(scaled_weights)

    return G


def G_to_w(G, max_weight, G_max):
    """Maps conductances onto weights.

    Parameters
    ----------
    G : ndarray
        Conductances in shape `m x 2n`.
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance.

    Returns
    ----------
    ndarray
        Synaptic weights of a single synaptic layer of size `m x n`.
    """
    scaled_weights = G_to_scaled_w(G)
    scaling_factor = max_weight/G_max
    weights = scaled_weights*scaling_factor

    return weights


def scaled_w_to_G(scaled_weights):
    """Maps scaled weights onto conductances.

    Parameters
    ----------
    scaled_weights : ndarray
        Scaled weights of shape `m x n`.

    Returns
    ----------
    ndarray
        Conductances in shape `m x 2n`.
    """
    # Positive and negative weights are implemented by two different sets
    # of conductances. Zero weights are implemented by leaving memristors 
    # unelectroformed.
    G = np.zeros((scaled_weights.shape[0], 2*scaled_weights.shape[1]))
    G_pos = np.where(scaled_weights > 0, scaled_weights, 0)
    G_neg = np.where(scaled_weights < 0, -scaled_weights, 0)
    # Odd columns dedicated to positive weights.
    G[:, 0::2] = G_pos
    # Even columns dedicated to negative weights.
    G[:, 1::2] = G_neg

    return G


def G_to_scaled_w(G):
    """Maps conductances onto scaled weights.

    Parameters
    ----------
    G : ndarray
        Conductances in shape `m x 2n`.

    Returns
    ----------
    ndarray
        Scaled weights of shape `m x n`.
    """
    scaled_weights = G[:, 0::2] - G[:, 1::2]

    return scaled_weights


def discretize_weights(weights, max_weight, distr, HRS_LRS, num_states, list_):
    """Discretizes weights.

    Parameters
    ----------
    weights : ndarray
        Synaptic weights of a single synaptic layer.
    max_weight : float
        Assumed maximum weight.
    distr : {'equal_G', 'equal_R', 'custom'}
        The spacing between states. 'equal_G' denotes equal spacing between
        conductance states, 'equal_R' denotes equal spacing between resistance
        states, and 'custom' denotes custom spacing between conductance states
        defined by argument `list_`.
    HRS_LRS : float
        Ratio G_min/G_max (equivalently R_max/R_min).
    num_states : float or int
        Number of states (including G_min and G_max).
    list_ : list of float
        Relative magnitudes of conductance levels if `distr == 'equal_G'`.

    Returns
    ----------
    ndarray
        Discretized weights.
    """
    if num_states != np.inf:
        weight_levels = levels.extract(
                type_=distr, ratio=HRS_LRS, num_states=num_states, list_=list_)
        weight_levels = np.array(weight_levels)*max_weight/weight_levels[-1]

        centers = (np.array(weight_levels)[1:] +
                   np.array(weight_levels)[:-1])/2
        # round to nearest discrete level
        indices = np.digitize(weights, centers)
        new_weights = np.array(weight_levels)[indices]
    else:
        new_weights = weights
        min_weight = max_weight/HRS_LRS
        mid_min_weight = min_weight/2
        new_weights = np.where(
            new_weights > max_weight, max_weight, new_weights)
        new_weights = np.where(
            new_weights < -max_weight, -max_weight, new_weights)
        new_weights = np.where(np.logical_and(
            new_weights < mid_min_weight,
            new_weights > -mid_min_weight),
            0, new_weights)
        new_weights = np.where(np.logical_and(
            new_weights > -min_weight,
            new_weights < -mid_min_weight),
            -min_weight, new_weights)
        new_weights = np.where(np.logical_and(
            new_weights < min_weight,
            new_weights > mid_min_weight),
            min_weight, new_weights)

    return new_weights
