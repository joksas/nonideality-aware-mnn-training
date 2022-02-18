from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import scipy.constants as const
import tensorflow as tf
import tensorflow_probability as tfp
from KDEpy import bw_selection
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow_probability import distributions as tfd

from awarememristor.crossbar import utils


class Nonideality(ABC):
    """Physical effect that influences the behavior of memristive devices."""

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
    """Nonideality whose effect can be simulated by disturbing the conductances."""

    @abstractmethod
    def disturb_G(self, G: tf.Tensor) -> tf.Tensor:
        """Disturb conductances."""


class LinearityNonpreserving(ABC):
    """Nonideality in which nonlinearity manifests itself in individual devices
    and the output current of a device is a function of its conductance
    parameter and the voltage applied across it."""

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

    @abstractmethod
    def k_V(self) -> float:
        """Return voltage scaling factor."""


class IVNonlinearityPF(Nonideality, LinearityNonpreserving):
    """Uses Poole-Frenkel model to compute currents."""

    def __init__(
        self,
        k_V: float,
        slopes: list[float],
        intercepts: list[float],
        res_cov_matrix: tf.Tensor,
    ) -> None:
        self.k_V_param = k_V
        self.slopes = slopes
        self.intercepts = intercepts
        self.res_cov_matrix = res_cov_matrix

    @staticmethod
    def model(V: tf.Tensor, c: tf.Tensor, d_times_perm: tf.Tensor) -> tf.Tensor:
        """Computes currents.

        Args:
            V: Voltages.
            c: Scaling factors associated with each of the conductances.
            d_times_perm: Products of thickness and permittivity associated
                with each of the conductances.

        Returns:
            Currents.
        """
        V_expanded = tf.expand_dims(V, axis=-1)
        return (
            c
            * V_expanded
            * tf.math.exp(
                const.elementary_charge
                * tf.math.sqrt(
                    const.elementary_charge * V_expanded / (const.pi * d_times_perm) + 1e-18
                )
                / (const.Boltzmann * (const.zero_Celsius + 20.0))
            )
        )

    @staticmethod
    def model_fitting(
        V: npt.NDArray[np.float64], c: float, d_times_perm: float
    ) -> npt.NDArray[np.float64]:
        """A helper function for fitting parameters using SciPy."""
        V = tf.constant(V)
        I = IVNonlinearityPF.model(V, c, d_times_perm)
        I = I.numpy()[:, 0]
        return I

    def compute_I(self, V, G):
        R = 1 / G

        ln_R = tf.math.log(R)

        fit_data = utils.multivariate_correlated_regression(
            ln_R, self.slopes, self.intercepts, self.res_cov_matrix
        )
        ln_c = fit_data[0]
        c = tf.math.exp(ln_c)
        ln_d_times_perm = fit_data[1]
        d_times_perm = tf.math.exp(ln_d_times_perm)

        I_ind = self.model(V, c, d_times_perm)

        I = utils.add_I_BL(I_ind)

        return I, I_ind

    def k_V(self):
        return self.k_V_param

    def label(self):
        return f"IVNL_PF:{self.slopes[0]:.3g}_{self.intercepts[0]:.3g}__{self.slopes[1]:.3g}_{self.intercepts[1]:.3g}"


class StuckAt(Nonideality, LinearityPreserving):
    """Models a fraction of the devices as stuck in one conductance state."""

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
    """Models a fraction of the devices as stuck at `G_off`."""

    def __init__(self, G_off: float, probability: float) -> None:
        StuckAt.__init__(self, G_off, probability)

    def label(self):
        return f"StuckOff:{self.probability:.3g}"


class StuckAtGOn(StuckAt):
    """Models a fraction of the devices as stuck at `G_on`."""

    def __init__(self, G_on: float, probability: float) -> None:
        StuckAt.__init__(self, G_on, probability)

    def label(self):
        return f"StuckOn:{self.probability:.3g}"


class StuckDistribution(Nonideality, LinearityPreserving):
    """Models a fraction of the devices as stuck at conductance states drawn
    from a random distribution.

    Kernel density estimation (KDE) with truncated normal distributions is
    constructed using a list of conductance values at which the devices got
    stuck.
    """

    def __init__(
        self, means: list[float], probability: float, bandwidth_def=bw_selection.scotts_rule
    ):
        """
        Args:
            means: Means of underlying normal distributions.
            probability: Probability that a given device will get stuck.
            bandwidth_def: Function used to determine the bandwidth parameter of KDE.
        """
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
            bandwidth: Standard deviation of underlying normal distributions.

        Returns:
            KDE distribution.
        """
        weights = []
        distr_means = []
        for mean in means:
            distr_means.append(mean)
            prob_neg = tfd.Normal(loc=mean, scale=bandwidth).cdf(0.0)
            # To ensure numerical stability, only include reflection if it will
            # have a non-negligible effect.
            if prob_neg > 1e-8:
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
    """Models D2D programming variability as lognormal deviations of resistances."""

    def __init__(
        self, G_off: float, G_on: float, R_on_log_std: float, R_off_log_std: float
    ) -> None:
        """
        Args:
            G_on: Memristor conductance in ON state.
            G_off: Memristor conductance in OFF state.
            R_on_log_std: Standard deviation of the (lognormal distribution's) underlying normal
                distribution associated with R_on (i.e. 1/G_on).
            R_off_log_std: Standard deviation of the (lognormal distribution's) underlying normal
                distribution associated with R_off (i.e. 1/G_off).
        """
        self.G_off = G_off
        self.G_on = G_on
        self.R_on_log_std = R_on_log_std
        self.R_off_log_std = R_off_log_std

    def label(self):
        return f"D2DLN:{self.R_on_log_std:.3g}_{self.R_off_log_std:.3g}"

    def disturb_G(self, G):
        R = 1 / G
        R_on = 1 / self.G_on
        R_off = 1 / self.G_off

        # Piece-wise linear interpolation.
        log_std_ref = [self.R_on_log_std, self.R_off_log_std]
        log_std = tfp.math.interp_regular_1d_grid(R, R_on, R_off, log_std_ref)

        # Lognormal modelling.
        R_squared = tf.math.pow(R, 2)
        log_var = tf.math.pow(log_std, 2)
        # Because $\sigma = \ln ( 1 + \frac{\sigma_X^2}{\mu_X^2} )$,
        # $\sigma_X^2 = \mu_X^2 (e^{\sigma^2} - 1)$.
        R_var = R_squared * (tf.math.exp(log_var) - 1.0)
        log_mu = tf.math.log(R_squared / tf.math.sqrt(R_squared + R_var))
        R = tfd.LogNormal(log_mu, log_std, validate_args=True).sample()

        G = 1 / R

        return G
