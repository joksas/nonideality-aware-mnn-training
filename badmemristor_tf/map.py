import numpy as np
import copy
import tensorflow as tf

from badmemristor_tf import levels


def w_to_G(weights, max_weight, G_min, G_max, distr="equal_G", num_states=np.inf, list_=None, scheme="proportional"):
    """Maps weights onto conductances.

    Parameters
    ----------
    weights : ndarray
        Synaptic weights of a single synaptic layer of size `m x n`.
    max_weight : float
        Assumed maximum weight.
    G_min : float
        Minimum conductance of electroformed memristors.
    G_max : float
        Maximum conductance of electroformed memristors.
    distr : {'equal_G', 'equal_R', 'custom'}, optional
        The spacing between states. 'equal_G' denotes equal spacing between
        conductance states, 'equal_R' denotes equal spacing between resistance
        states, and 'custom' denotes custom spacing between conductance states
        defined by argument `list_`.
    num_states : float or int, optional
        Number of states (including G_min and G_max).
    list_ : list of float, optional
        Relative magnitudes of conductance levels if `distr == 'custom'`.
    scheme : {"proportional", "differential"}, optional
        Mapping scheme.

    Returns
    ----------
    ndarray
        Conductances in shape `m x 2n`.
    """
    G_eff = w_to_G_eff(weights, max_weight, G_min, G_max, distr=distr,
            num_states=num_states, list_=list_, scheme=scheme)
    G = G_eff_to_G(G_eff, G_min=G_min, scheme=scheme)

    return G


def w_to_G_eff(weights, max_weight, G_min, G_max, distr="equal_G", num_states=np.inf, list_=None, scheme="proportional"):
    """Maps weights onto effective conductances.

    Parameters
    ----------
    weights : ndarray
        Synaptic weights of a single synaptic layer of size `m x n`.
    max_weight : float
        Assumed maximum weight.
    G_min : float
        Minimum conductance of electroformed memristors.
    G_max : float
        Maximum conductance of electroformed memristors.
    distr : {'equal_G', 'equal_R', 'custom'}, optional
        The spacing between states. 'equal_G' denotes equal spacing between
        conductance states, 'equal_R' denotes equal spacing between resistance
        states, and 'custom' denotes custom spacing between conductance states
        defined by argument `list_`.
    num_states : float or int, optional
        Number of states (including G_min and G_max).
    list_ : list of float, optional
        Relative magnitudes of conductance levels if `distr == 'custom'`.
    scheme : {"proportional", "differential"}, optional
        Mapping scheme.

    Returns
    ----------
    ndarray
        Effective conductances of shape `m x n`.
    """
    HRS_LRS = G_max/G_min
    if scheme == "proportional":
        weights = discretize_weights(
                weights, max_weight, distr, HRS_LRS, num_states, list_)
    elif scheme == "differential":
        # TODO: Implement discretization for general differential pair scheme. Maybe consider all possible pairs of conductances and make a tradeoff between closeness to desired state and low conductance?
        weights = clip_max_weights(weights, max_weight)

    k_G = compute_k_G(max_weight, G_max, G_min=G_min, scheme=scheme)
    G_eff = k_G*weights

    return G_eff


def G_to_w(G, max_weight, G_max, G_min=None, scheme="proportional"):
    """Maps conductances onto weights.

    Parameters
    ----------
    G : ndarray
        Conductances in shape `m x 2n`.
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance.
    G_min : float, optional
        Minimum conductance of electroformed memristors. Necessary to provide
        if `scheme == "proportional"`.
    scheme : {"proportional", "differential"}, optional
        Mapping scheme.

    Returns
    ----------
    ndarray
        Synaptic weights of a single synaptic layer of size `m x n`.
    """
    G_eff = G_to_G_eff(G)
    weights = G_eff_to_w(G_eff, max_weight, G_max, scheme=scheme)

    return weights


def G_eff_to_w(G_eff, max_weight, G_max, G_min=None, scheme="proportional"):
    """Maps effective conductances onto weights.

    Parameters
    ----------
    G_eff : ndarray
        Effective conductances of shape `m x n`.
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance.
    G_min : float, optional
        Minimum conductance of electroformed memristors. Necessary to provide
        if `scheme == "proportional"`.
    scheme : {"proportional", "differential"}, optional
        Mapping scheme.

    Returns
    ----------
    ndarray
        Synaptic weights of a single synaptic layer of size `m x n`.
    """
    k_G = compute_k_G(max_weight, G_max, G_min=G_min, scheme=scheme)
    weights = G_eff/k_G

    return weights

def I_to_y(I, k_V, max_weight, G_max, G_min, scheme="proportional"):
    """Converts output currents of a dot-product engine onto synaptic layer inputs.

    Parameters
    ----------
    I : ndarray
        Output currents of shape (p x 2n)
    k_V : float
        Voltage scaling factor.
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance of electroformed memristors.
    G_min : float, optional
        Minimum conductance of electroformed memristors. Necessary to provide
        if `scheme == "proportional"`.
    scheme : {"proportional", "differential"}, optional
        Mapping scheme.

    Returns
    ----------
    y : ndarray
        Outputs of shape (p x n) of a synaptic layer implemented using
        memristive crossbars.
    """
    I_total = I[:, 0::2] - I[:, 1::2]
    y = I_total_to_y(I_total, k_V, max_weight, G_max, G_min, scheme=scheme)
    return y


def I_total_to_y(I_total, k_V, max_weight, G_max, G_min, scheme="proportional"):
    """Converts total output currents of a dot-product engine onto synaptic layer
    inputs.

    Parameters
    ----------
    I_total : ndarray
        Total output currents of shape (p x n)
    k_V : float
        Voltage scaling factor.
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance of electroformed memristors.
    G_min : float, optional
        Minimum conductance of electroformed memristors. Necessary to provide
        if `scheme == "proportional"`.
    scheme : {"proportional", "differential"}, optional
        Mapping scheme.

    Returns
    ----------
    y : ndarray
        Outputs of shape (p x n) of a synaptic layer implemented using
        memristive crossbars.
    """
    k_G = compute_k_G(max_weight, G_max, G_min, scheme=scheme)
    k_I = compute_k_I(k_V, k_G)
    y = I_total/k_I
    return y


def G_eff_to_G(G_eff, G_min=None, scheme="proportional"):
    """Maps effective weights onto conductances.

    Parameters
    ----------
    G_eff : ndarray
        Effective conductances of shape `m x n`.
    G_min : float, optional
        Minimum conductance of electroformed memristors. Necessary to provide
        if `scheme == "proportional"`.
    scheme : {"proportional", "differential"}, optional
        Mapping scheme.

    Returns
    ----------
    ndarray
        Conductances in shape `m x 2n`.
    """
    # Positive and negative weights are implemented by two different sets of
    # conductances.

    if scheme == "proportional":
        # Zero weights are implemented by leaving memristors unelectroformed.
        G_pos = tf.where(G_eff > 0, G_eff, 0.0)
        G_neg = tf.where(G_eff < 0, -G_eff, 0.0)
    elif scheme == "differential":
        G_eff_abs = tf.math.abs(G_eff)
        # We implement the pairs by choosing the lowest possible conductances.
        G_pos = tf.math.maximum(G_eff, 0.0) + G_min
        G_neg = -tf.math.minimum(G_eff, 0.0) + G_min

    # Odd columns dedicated to positive weights.
    # Even columns dedicated to negative weights.
    G = tf.reshape(
        tf.concat([G_pos[...,tf.newaxis], G_neg[...,tf.newaxis]], axis=-1), 
        [tf.shape(G_pos)[0],-1])

    return G


def G_to_G_eff(G):
    """Maps conductances onto effective conductances.

    Parameters
    ----------
    G : ndarray
        Conductances in shape `m x 2n`.

    Returns
    ----------
    ndarray
        Effective conductances of shape `m x n`.
    """
    G_eff = G[:, 0::2] - G[:, 1::2]

    return G_eff


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
        Relative magnitudes of conductance levels if `distr == 'custom'`.

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
        # If the weights cannot be implemented precisely, they are rounded to
        # the nearest available level. If weight > max_weight, it is rounded to
        # max_weight, if it min_weight/2 < weight < min_weight, it is rounded
        # to min_weight, and if 0 < weight <= min_weight/2, it is rounded to 0.
        # Similarly for negative weights.
        new_weights = clip_max_weights(weights, max_weight)
        min_weight = max_weight/HRS_LRS
        mid_min_weight = min_weight/2
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


def clip_max_weights(weights, max_weight):
    """Clips weights below -max_weight and above max_weight.

    Parameters
    ----------
    weights : ndarray
        Synaptic weights.
    max_weight : float
        Assumed maximum weight.

    Returns
    ----------
    new_weights : ndarray
        Clipped weights.
    """
    #weights = tf.clip_by_value(weights, -max_weight, max_weight)
    weights = tf.clip_by_value(weights, 0.0, max_weight)

    return weights


def compute_k_G(max_weight, G_max, G_min, scheme="proportional"):
    """Computes conductance scaling factor.

    Parameters
    ----------
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance of electroformed memristors.
    G_min : float, optional
        Minimum conductance of electroformed memristors. Necessary to provide
        if `scheme == "proportional"`.
    scheme : {"proportional", "differential"}, optional
        Mapping scheme.

    Returns
    ----------
    float
        Conductance scaling factor.
    """
    if scheme == "proportional":
        k_G = G_max/max_weight
    elif scheme == "differential":
        k_G = (G_max-G_min)/max_weight

    return k_G


def compute_k_I(k_V, k_G):
    """Computes current scaling factor.

    Parameters
    ----------
    k_V : float
        Voltage scaling factor.
    k_G : float
        Conductance scaling factor.

    Returns
    ----------
    float
        Current scaling factor.
    """
    return k_V*k_G


def x_to_V(x, k_V):
    """Maps inputs (to a synaptic layer) onto voltages.

    Parameters
    ----------
    x : ndarray
        Synaptic inputs.
    k_V : float
        Voltage scaling factor.

    Returns
    ----------
    ndarray
        Voltages.
    """
    return k_V*x

