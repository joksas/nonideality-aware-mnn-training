import numpy as np
import tensorflow as tf
from KDEpy import bw_selection
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow_probability import distributions as tfd

from . import utils


class StuckDistribution:
    def __init__(
        self, means: list[float], probability: float, bandwidth_def=bw_selection.scotts_rule
    ):
        bandwidth = bandwidth_def(np.reshape(means, (len(means), 1)))
        self.__probability = probability
        self.__distribution = self._kde(means, bandwidth)

    @staticmethod
    def _kde(means: list[float], bandwidth: float) -> tfd.Distribution:
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

    def disturb_G(self, G: tf.Tensor) -> tf.Tensor:
        mask = utils.random_bool_tensor(G.shape, self.__probability)
        idxs = tf.where(mask)
        zeroed_G = tf.where(mask, 0.0, G)
        stuck_G = self.__distribution.sample(len(idxs))
        disturbed_G = zeroed_G + tf.scatter_nd(idxs, stuck_G, G.shape)
        return disturbed_G
