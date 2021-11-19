import math
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from awarememristor.crossbar import utils
from KDEpy import bw_selection
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow_probability import distributions as tfd


class Nonideality(ABC):
    @abstractmethod
    def label(self) -> str:
        """Returns nonideality label used in directory names, for example."""

    def __eq__(self, other):
        if self is None or other is None:
            if self is None and other is None:
                return True
            return False
        return self.label() == other.label()


class LinearityPreserving(ABC):
    @abstractmethod
    def disturb_G(self, G: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Disturb conductances."""


class LinearityNonpreserving(ABC):
    @abstractmethod
    def compute_I(self, V: tf.Tensor, G: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute currents in a crossbar suffering from linearity-nonpreserving nonideality.

        Args:
            V: Voltages of shape `p x m`.
            G: Conductances of shape `m x n`.

        Returns:
            I: Output currents of shape `p x n`.
            I_ind: Currents of shape `p x m x n` produced by each of the
                conductances in the crossbar array.
        """


class IVNonlinearity(Nonideality, LinearityNonpreserving):
    def __init__(self, V_ref: float, n_avg: float, n_std: float) -> None:
        self.V_ref = V_ref
        self.n_avg = n_avg
        self.n_std = n_std

    def label(self):
        return f"IVNL:{self.n_avg:.3g}_{self.n_std:.3g}"

    def compute_I(self, V, G):
        """Compute current values by modelling I-V behaviour using nonlinearity
        parameter.

        Nonlinearity parameter `n` is defined as the current generated by a
        resistive device at voltage `2*V` divided by the current generated by
        the device at voltage `V`. We introduce voltage `V_ref` at which the
        generated amount of current equals the expected amount described by
        Ohm's law, i.e. `I = V*G`.
        """
        n = tf.random.normal(G.get_shape().as_list(), mean=self.n_avg, stddev=self.n_std)
        # n <= 1 would produce unrealistic behaviour, while 1 < n < 2 is not typical in I-V curves
        n = tf.clip_by_value(n, 2.0, math.inf)

        ohmic_current = self.V_ref * tf.expand_dims(G, axis=0)
        # Take absolute value of V to prevent negative numbers from being raised to
        # a negative power. We assume symmetrical behaviour with negative voltages.
        ratio = tf.expand_dims(tf.abs(V) / self.V_ref, axis=-1)
        exponent = utils.tf_log2(n)
        sign = tf.expand_dims(tf.sign(V), axis=-1)

        I_ind = sign * ohmic_current * ratio ** exponent

        I = utils.add_I_BL(I_ind)

        return I, I_ind


class StuckAt(Nonideality, LinearityPreserving):
    def __init__(self, value: float, probability: float) -> None:
        """
        Args:
            value: Conductance value to set randomly selected devices to.
            probability: Probability that a given device will be set to `val`.
                Probability must be in the [0.0, 1.0] range.
        """
        self.value = value
        self.probability = probability

    def label(self):
        return f"Stuck:{self.value:.3g}_{self.probability:.3g}"

    def disturb_G(self, G):
        mask = utils.random_bool_tensor(G.shape, self.probability)
        G = tf.where(mask, self.value, G)
        return G


class StuckAtGOff(StuckAt):
    def __init__(self, G_off: float, probability: float) -> None:
        StuckAt.__init__(self, G_off, probability)

    def label(self):
        return f"StuckOff:{self.probability:.3g}"


class StuckAtGOn(StuckAt):
    def __init__(self, G_on: float, probability: float) -> None:
        StuckAt.__init__(self, G_on, probability)

    def label(self):
        return f"StuckOn:{self.probability:.3g}"


class StuckDistribution(Nonideality, LinearityPreserving):
    def __init__(
        self, means: list[float], probability: float, bandwidth_def=bw_selection.scotts_rule
    ):
        bandwidth = bandwidth_def(np.reshape(means, (len(means), 1)))
        self.probability = probability
        self.bandwidth = bandwidth
        self.distribution = self._kde(means, bandwidth)

    def label(self) -> str:
        return f"StuckDistr:{self.probability:.3g}_{self.bandwidth:.3g}"

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

    def disturb_G(self, G):
        mask = utils.random_bool_tensor(G.shape, self.probability)
        idxs = tf.where(mask)
        zeroed_G = tf.where(mask, 0.0, G)
        stuck_G = self.distribution.sample(tf.math.count_nonzero(mask))
        disturbed_G = zeroed_G + tf.scatter_nd(idxs, stuck_G, G.shape)
        return disturbed_G


class D2DLognormal(Nonideality, LinearityPreserving):
    def __init__(self, G_off: float, G_on: float, R_on_std: float, R_off_std: float) -> None:
        """
        Args:
            R_on_std: Standard deviation of the (lognormal distribution's) underlying normal
                distribution associated with R_on (i.e. 1/G_on).
            R_off_std: Standard deviation of the (lognormal distribution's) underlying normal
                distribution associated with R_off (i.e. 1/G_off).
        """
        self.G_off = G_off
        self.G_on = G_on
        self.R_on_std = R_on_std
        self.R_off_std = R_off_std

    def label(self):
        return f"D2DLN:{self.R_on_std:.3g}_{self.R_off_std:.3g}"

    def disturb_G(self, G):
        """Disturb conductances lognormally."""
        R = 1 / G
        R_on = 1 / self.G_on
        R_off = 1 / self.G_off

        # Piece-wise linear interpolation
        std_ref = [self.R_on_std, self.R_off_std]
        R_std = tfp.math.interp_regular_1d_grid(R, R_on, R_off, std_ref)

        # Lognormal modelling
        R2 = tf.math.pow(R, 2)
        R_std2 = tf.math.pow(R_std, 2)
        R_mu = tf.math.log(R2 / tf.math.sqrt(R2 + R_std2))
        R = tfd.LogNormal(R_mu, R_std, validate_args=True).sample()

        G = 1 / R

        return G
