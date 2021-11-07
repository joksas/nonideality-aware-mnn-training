import numpy as np
import tensorflow as tf

from . import utils


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
    mask = utils.random_bool_tensor(G.shape, prob)
    G = tf.where(mask, val, G)
    return G
