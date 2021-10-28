import tensorflow as tf


def tf_log2(x: tf.Tensor):
    """Computes logarithm of base 2 of x.

    x : tf. constant

    Returns
    ----------
    tf.constant
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def add_I_BL(I_ind: tf.Tensor):
    """Adds currents along the bit lines.

    Parameters
    ----------
    I_ind : ndarray
        Currents of shape `p x m x n` produced by each of the conductances in
        the crossbar array.

    Returns
    ----------
    I : ndarray
        Output currents of shape `p x n`.
    """
    I = tf.math.reduce_sum(I_ind, axis=1)
    return I
