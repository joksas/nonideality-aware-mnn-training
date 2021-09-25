import tensorflow as tf


def random_bool_tensor(shape, prob_true):
    """Returns random boolean tensor.

    Parameters
    ----------
    shape : list of int
        Tensor shape.
    prob_true : float
        Probability that a given entry is going to be True. Probability must be in the [0.0, 1.0]
        range.
    """
    random_float_tensor = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.dtypes.float64)
    return random_float_tensor < prob_true
