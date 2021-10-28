import tensorflow as tf


def tf_log2(x: tf.Tensor) -> tf.Tensor:
    """Compute logarithm of base 2 of `x`.

    Args:
        x: An array.
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def add_I_BL(I_ind: tf.Tensor) -> tf.Tensor:
    """Add currents along the bit lines.

    Args:
        I_ind: Currents of shape `p x m x n` produced by each of the conductances in the crossbar
            array.

    Returns:
        Output currents of shape `p x n`.
    """
    I = tf.math.reduce_sum(I_ind, axis=1)
    return I
