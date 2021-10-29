import tensorflow as tf


def I_to_y(I: tf.Tensor, k_V: float, max_weight: float, G_max: float, G_min: float) -> tf.Tensor:
    """Convert output currents of a dot-product engine onto synaptic layer inputs.

    Args:
        I: Output currents of shape `p x 2n`
        k_V: Voltage scaling factor.
        max_weight: Assumed maximum weight.
        G_max: Maximum conductance of electroformed memristors.
        G_min: Minimum conductance of electroformed memristors.

    Returns:
        Outputs of shape `p x n` of a synaptic layer implemented using memristive crossbars.
    """
    I_total = I[:, 0::2] - I[:, 1::2]
    y = I_total_to_y(I_total, k_V, max_weight, G_max, G_min)
    return y


def I_total_to_y(
    I_total: tf.Tensor, k_V: float, max_weight: float, G_max: float, G_min: float
) -> tf.Tensor:
    """Convert total output currents of a dot-product engine onto synaptic layer inputs.

    Args:
        I_total: Total output currents of shape `p x n`
        k_V: Voltage scaling factor.
        max_weight: Assumed maximum weight.
        G_max: Maximum conductance of electroformed memristors.
        G_min: Minimum conductance of electroformed memristors.

    Returns:
        Outputs of shape `p x n` of a synaptic layer implemented using memristive crossbars.
    """
    k_G = compute_k_G(max_weight, G_max, G_min)
    k_I = compute_k_I(k_V, k_G)
    y = I_total / k_I
    return y


def clip_weights(weights: tf.Tensor, max_weight: float) -> tf.Tensor:
    """Clip weights below 0 and above `max_weight`.

    Args:
        weights: Synaptic weights.
        max_weight: Assumed maximum weight.

    Returns:
        Clipped weights.
    """
    weights = tf.clip_by_value(weights, 0.0, max_weight)

    return weights


def compute_k_G(max_weight: float, G_max: float, G_min: float) -> float:
    """Compute conductance scaling factor.

    Args:
        max_weight: Assumed maximum weight.
        G_max: Maximum conductance of electroformed memristors.
        G_min: Minimum conductance of electroformed memristors.

    Returns:
        Conductance scaling factor.
    """
    k_G = (G_max - G_min) / max_weight

    return k_G


def compute_k_I(k_V: float, k_G: float) -> float:
    """Compute current scaling factor.

    Args:
        k_V: Voltage scaling factor.
        k_G: Conductance scaling factor.

    Returns:
        Current scaling factor.
    """
    return k_V * k_G


def x_to_V(x: tf.Tensor, k_V: float) -> tf.Tensor:
    """Map inputs (to a synaptic layer) onto voltages.

    Args:
        x: Synaptic inputs.
        k_V: Voltage scaling factor.

    Returns:
        Voltages.
    """
    return k_V * x


def w_params_to_G(weight_params: tf.Tensor, G_min: float, G_max: float) -> tf.Tensor:
    """Map weight parameters onto conductances.

    Args:
        weight_params: Weight parameters of shape `m x 2n`. These are used to train each conductance
            (instead of pair of conductances) directly.
        G_min: Minimum conductance of electroformed memristors.
        G_max: Maximum conductance of electroformed memristors.

    Returns:
        G: Conductances of shape `m x 2n`.
        max_weight: Assumed maximum weight.
    """
    max_weight = tf.math.reduce_max(weight_params)

    weight_params = clip_weights(weight_params, max_weight)

    k_G = compute_k_G(max_weight, G_max, G_min)
    G = k_G * weight_params + G_min

    return G, max_weight


def w_to_G(weights: tf.Tensor, G_min: float, G_max: float) -> tf.Tensor:
    """Map weights onto conductances.

    Args:
        weights: Weights of shape `m x n`.
        G_min: Minimum conductance of electroformed memristors.
        G_max: Maximum conductance of electroformed memristors.

    Returns:
        G: Conductances of shape `m x n`.
        max_weight: Assumed maximum weight.
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
