import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow_probability import distributions as tfd


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


def kde(means: list[float], bandwidth: float) -> tfd.Distribution:
    """Kernel density estimation.

    Args:
        means: Means of underlying normal distributions.
        width: Standard deviation of underlying normal distributions.

    Returns:
        KDE distribution.
    """
    weights = []
    distr_means = []
    for mean in means:
        distr_means.append(mean)
        prob_neg = tfd.Normal(loc=mean, scale=bandwidth).cdf(0.0)
        if prob_neg > 1e-8:  # Only include reflection if numerically stable.
            distr_means.append(-mean)
            prob_pos = 1.0 - prob_neg
            weights.extend([prob_pos, prob_neg])
        else:
            weights.append(1.0)

    np_config.enable_numpy_behavior()
    distr_means_32 = tf.constant(distr_means)
    bandwidth_32 = tf.constant(bandwidth)

    kde_distribution = tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights),
        tfd.TruncatedNormal(
            loc=distr_means_32.astype(tf.float32),
            scale=bandwidth_32.astype(tf.float32),
            low=0.0,
            high=np.inf,
        ),
    )

    return kde_distribution
