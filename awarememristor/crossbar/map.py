import tensorflow as tf


def I_to_y(I: tf.Tensor, k_V: float, max_weight: float, G_on: float, G_off: float) -> tf.Tensor:
    """Convert output currents of a dot-product engine onto synaptic layer inputs.

    Args:
        I: Output currents of shape `p x 2n`
        k_V: Voltage scaling factor.
        max_weight: Assumed maximum weight.
        G_on: Memristor conductance in ON state.
        G_off: Memristor conductance in OFF state.

    Returns:
        Outputs of shape `p x n` of a synaptic layer implemented using memristive crossbars.
    """
    I_total = I[:, 0::2] - I[:, 1::2]
    y = I_total_to_y(I_total, k_V, max_weight, G_on, G_off)
    return y


def I_total_to_y(
    I_total: tf.Tensor, k_V: float, max_weight: float, G_on: float, G_off: float
) -> tf.Tensor:
    """Convert total output currents of a dot-product engine onto synaptic layer inputs.

    Args:
        I_total: Total output currents of shape `p x n`
        k_V: Voltage scaling factor.
        max_weight: Assumed maximum weight.
        G_on: Memristor conductance in ON state.
        G_off: Memristor conductance in OFF state.

    Returns:
        Outputs of shape `p x n` of a synaptic layer implemented using memristive crossbars.
    """
    k_G = compute_k_G(max_weight, G_on, G_off)
    k_I = compute_k_I(k_V, k_G)
    y = I_total / k_I
    return y


def compute_k_G(max_weight: float, G_on: float, G_off: float) -> float:
    """Compute conductance scaling factor.

    Args:
        max_weight: Assumed maximum weight.
        G_on: Memristor conductance in ON state.
        G_off: Memristor conductance in OFF state.

    Returns:
        Conductance scaling factor.
    """
    k_G = (G_on - G_off) / max_weight

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


def w_params_to_G(weight_params: tf.Tensor, G_off: float, G_on: float) -> tf.Tensor:
    """Map weight parameters onto conductances.

    Args:
        weight_params: Nonnegative weight parameters of shape `m x 2n`. These are used to train
            each conductance (instead of pair of conductances) directly.
        G_off: Memristor conductance in OFF state.
        G_on: Memristor conductance in ON state.

    Returns:
        G: Conductances of shape `m x 2n`.
        max_weight: Assumed maximum weight.
    """
    max_weight = tf.math.reduce_max(weight_params)
    k_G = compute_k_G(max_weight, G_on, G_off)
    G = k_G * weight_params + G_off

    return G, max_weight


def w_to_G(weights: tf.Tensor, G_off: float, G_on: float) -> tf.Tensor:
    """Map weights onto conductances.

    Args:
        weights: Weights of shape `m x n`.
        G_off: Memristor conductance in OFF state.
        G_on: Memristor conductance in ON state.

    Returns:
        G: Conductances of shape `m x n`.
        max_weight: Assumed maximum weight.
    """
    max_weight = tf.math.reduce_max(tf.math.abs(weights))

    k_G = compute_k_G(max_weight, G_on, G_off)
    G_eff = k_G * weights

    # We implement the pairs by choosing the lowest possible conductances.
    G_pos = tf.math.maximum(G_eff, 0.0) + G_off
    G_neg = -tf.math.minimum(G_eff, 0.0) + G_off

    # Odd columns dedicated to positive weights.
    # Even columns dedicated to negative weights.
    G = tf.reshape(
        tf.concat([G_pos[..., tf.newaxis], G_neg[..., tf.newaxis]], axis=-1),
        [tf.shape(G_pos)[0], -1],
    )

    return G, max_weight
