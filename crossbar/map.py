import tensorflow as tf


def I_to_y(I, k_V, max_weight, G_max, G_min):
    """Converts output currents of a dot-product engine onto synaptic layer inputs.

    Parameters
    ----------
    I : ndarray
        Output currents of shape `p x 2n`
    k_V : float
        Voltage scaling factor.
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance of electroformed memristors.
    G_min : float
        Minimum conductance of electroformed memristors.

    Returns
    ----------
    y : ndarray
        Outputs of shape `p x n` of a synaptic layer implemented using
        memristive crossbars.
    """
    I_total = I[:, 0::2] - I[:, 1::2]
    y = I_total_to_y(I_total, k_V, max_weight, G_max, G_min)
    return y


def I_total_to_y(I_total, k_V, max_weight, G_max, G_min):
    """Converts total output currents of a dot-product engine onto synaptic layer
    inputs.

    Parameters
    ----------
    I_total : ndarray
        Total output currents of shape `p x n`
    k_V : float
        Voltage scaling factor.
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance of electroformed memristors.
    G_min : float, optional
        Minimum conductance of electroformed memristors.

    Returns
    ----------
    y : ndarray
        Outputs of shape `p x n` of a synaptic layer implemented using
        memristive crossbars.
    """
    k_G = compute_k_G(max_weight, G_max, G_min)
    k_I = compute_k_I(k_V, k_G)
    y = I_total / k_I
    return y


def clip_weights(weights, max_weight):
    """Clips weights below 0 and above max_weight.

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
    weights = tf.clip_by_value(weights, 0.0, max_weight)

    return weights


def compute_k_G(max_weight, G_max, G_min):
    """Computes conductance scaling factor.

    Parameters
    ----------
    max_weight : float
        Assumed maximum weight.
    G_max : float
        Maximum conductance of electroformed memristors.
    G_min : float, optional
        Minimum conductance of electroformed memristors.

    Returns
    ----------
    float
        Conductance scaling factor.
    """
    k_G = (G_max - G_min) / max_weight

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
    return k_V * k_G


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
    return k_V * x


def w_params_to_G(weight_params, G_min, G_max):
    """Maps weight parameters onto conductances.

    Parameters
    ----------
    weight_params : ndarray
        Weight parameters of shape `m x 2n`. These are used to
        train each conductance (instead of pair of conductances)
        directly.
    G_min : float
        Minimum conductance of electroformed memristors.
    G_max : float
        Maximum conductance of electroformed memristors.

    Returns
    ----------
    G : ndarray
        Conductances of shape `m x 2n`.
    max_weight : float
        Assumed maximum weight.
    """
    max_weight = tf.math.reduce_max(weight_params)

    weight_params = clip_weights(weight_params, max_weight)

    k_G = compute_k_G(max_weight, G_max, G_min)
    G = k_G * weight_params + G_min

    return G, max_weight


def w_to_G(weights, G_min, G_max):
    """Maps weights onto conductances.

    Parameters
    ----------
    weight : ndarray
        Weights of shape `m x n`.
    G_min : float
        Minimum conductance of electroformed memristors.
    G_max : float
        Maximum conductance of electroformed memristors.

    Returns
    ----------
    G : ndarray
        Conductances of shape `m x n`.
    max_weight : float
        Assumed maximum weight.
    """
    max_weight = tf.math.reduce_max(tf.math.abs(weights))

    k_G = compute_k_G(max_weight, G_max, G_min)
    G_eff = k_G * weights

    # We implement the pairs by choosing the lowest possible conductances.
    G_pos = tf.math.maximum(G_eff, 0.0) + G_min
    G_neg = -tf.math.minimum(G_eff, 0.0) + G_min

    # Odd columns dedicated to positive weights.
    # Even columns dedicated to negative weights.
    G = tf.reshape(
        tf.concat([G_pos[..., tf.newaxis], G_neg[..., tf.newaxis]], axis=-1),
        [tf.shape(G_pos)[0], -1],
    )

    return G, max_weight
