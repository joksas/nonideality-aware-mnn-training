import tensorflow as tf
import tensorflow_probability as tfp


def lognormal(G, G_min, G_max, R_min_std, R_max_std):
    """Disturbs conductances lognormally.

    Parameters
    ----------
    G : tf.Tensor
        Conductances.
    G_min : float
        Conductance associated with the OFF state.
    G_max : float
        Conductance associated with the ON state.
    R_min_std : float
        Standard deviation of the (lognormal distribution's) underlying normal distribution
        associated with R_min (i.e. 1/G_max).
    R_max_std : float
        Standard deviation of the (lognormal distribution's) underlying normal distribution
        associated with R_max (i.e. 1/G_min).
    """
    R = 1 / G
    R_min = 1 / G_max
    R_max = 1 / G_min

    # Piece-wise linear interpolation
    std_ref = [R_min_std, R_max_std]
    R_std = tfp.math.interp_regular_1d_grid(R, R_min, R_max, std_ref)

    # Lognormal modelling
    R2 = tf.math.pow(R, 2)
    R_std2 = tf.math.pow(R_std, 2)
    R_mu = tf.math.log(R2 / tf.math.sqrt(R2 + R_std2))
    R = tfp.distributions.LogNormal(R_mu, R_std, validate_args=True).sample()

    G = 1 / R

    return G
