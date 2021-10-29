import tensorflow as tf


def random_devices_stuck(G: tf.Tensor, val: float, prob: float) -> tf.Tensor:
    """Set random elements of `G` to `val`.

    Args:
        G: Conductances.
        val: Conductance value to set randomly selected devices to.
        prob: Probability that a given device will be set to `val`. Probability must be in the
            [0.0, 1.0] range.

    Returns:
        Modified conductances.
    """
    mask = random_bool_tensor(G.shape, prob)
    G = tf.where(mask, val, G)
    return G


def random_bool_tensor(shape: list[int], prob_true: float) -> tf.Tensor:
    """Return random boolean tensor.

    Args:
        shape: Tensor shape.
        prob_true: Probability that a given entry is going to be True. Probability must be in the
            [0.0, 1.0] range.
    """
    random_float_tensor = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.dtypes.float64)
    return random_float_tensor < prob_true
