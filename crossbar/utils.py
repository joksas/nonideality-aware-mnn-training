import tensorflow as tf


def tf_log2(x):
    """Computes logarithm of base 2 of x.

    x : tf. constant

    Returns
    ----------
    tf.constant
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator/denominator
