import tensorflow as tf


def random_devices_stuck(G, val, prob):
    """Sets random elements of G to val.

    Parameters
    ----------
    G : tf.Tensor
        Conductances.
    val : float
        Conductances to set randomly selected devices to.
    prob : float
        Probability that a given device will be set to val. Probability must be in the [0.0, 1.0]
        range.
    """
    mask = random_bool_tensor(G.shape, prob)
    G = tf.where(mask, val, G)
    return G


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
