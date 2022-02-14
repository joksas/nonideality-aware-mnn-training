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


def random_bool_tensor(shape: list[int], prob_true: float) -> tf.Tensor:
    """Return random boolean tensor.

    Args:
        shape: Tensor shape.
        prob_true: Probability that a given entry is going to be True. Probability must be in the
            [0.0, 1.0] range.
    """
    random_float_tensor = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.dtypes.float64)
    return random_float_tensor < prob_true


def linear_fit(x: tf.Tensor, slope: float, intercept: float, res_std: float):
    """Return linear fit with random normal deviations.

    Args:
        x: Input tensor.
        slope: Slope of the linear fit.
        intercept: Intercept of the linear fit.
        res_std: Standard deviation of the random normal deviations.
    """
    fit = slope * x + intercept
    deviations = tf.random.normal(x.shape, mean=0.0, stddev=res_std)
    return fit + deviations
