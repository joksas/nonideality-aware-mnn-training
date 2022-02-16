import numpy as np
import pytest
import tensorflow as tf

from awarememristor.crossbar import nonidealities
from tests import utils

# Only special case, i.e. when std = 0.0 for all entries.
d2d_lognormal_testdata = [
    (
        (
            0.5,
            0.6,
            0.0,
            0.0,
            tf.constant(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
        ),
        tf.constant(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    ),
    (
        (
            1.0,
            6.0,
            0.0,
            0.0,
            tf.constant(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
        ),
        tf.constant(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize("args,expected", d2d_lognormal_testdata)
def test_d2d_lognormal(args, expected):
    G_off, G_on, R_on_log_std, R_off_log_std, G = args
    nonideality = nonidealities.D2DLognormal(G_off, G_on, R_on_log_std, R_off_log_std)
    result = nonideality.disturb_G(G)
    utils.assert_tf_approx(result, expected)


iv_nonlinearity_pf_I_ind_testdata = [
    (
        (
            tf.constant(
                [
                    [1.0, 0.0, -0.5],
                    [0.0, 0.25, 0.0],
                ]
            ),
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
            tf.constant(
                [
                    [np.inf, np.inf, np.inf, np.inf],
                    [np.inf, np.inf, np.inf, np.inf],
                    [np.inf, np.inf, np.inf, np.inf],
                ]
            ),
        ),
        # With `d_times_perm = np.inf`, currents should be produced
        # according to Ohm's law with `c` acting as conductance.
        tf.constant(
            [
                [
                    [1.0 * 1.0, 1.0 * 2.0, 1.0 * 3.0, 1.0 * 4.0],
                    [0.0 * 5.0, 0.0 * 6.0, 0.0 * 7.0, 0.0 * 8.0],
                    [-0.5 * 9.0, -0.5 * 10.0, -0.5 * 11.0, -0.5 * 12.0],
                ],
                [
                    [0.0 * 1.0, 0.0 * 2.0, 0.0 * 3.0, 0.0 * 4.0],
                    [0.25 * 5.0, 0.25 * 6.0, 0.25 * 7.0, 0.25 * 8.0],
                    [0.0 * 9.0, 0.0 * 10.0, 0.0 * 11.0, 0.0 * 12.0],
                ],
            ]
        ),
    ),
    (
        (
            tf.constant(
                [1.0, 0.0, -0.5],
            ),
            0.5,
            np.inf,
        ),
        # With `d_times_perm = np.inf`, currents should be produced
        # according to Ohm's law with `c` acting as conductance.
        tf.constant(
            [
                [0.5],
                [0.0],
                [-0.25],
            ]
        ),
    ),
]


@pytest.mark.parametrize("args,expected", iv_nonlinearity_pf_I_ind_testdata)
def test_iv_nonlinearity_pf_I_ind(args, expected):
    I_ind_exp = expected
    V, c, d_times_perm = args
    I_ind = nonidealities.IVNonlinearityPF.model(V, c, d_times_perm)
    utils.assert_tf_approx(I_ind, I_ind_exp)


iv_nonlinearity_pf_I_testdata = [
    (
        (
            nonidealities.IVNonlinearityPF(
                0.5,
                (-1, -2, 0),
                (1, 1, 0),
            ),
            tf.constant(
                [
                    [1.0, 0.0],
                    [0.0, -0.25],
                    [1.0, 0.0],
                ]
            ),
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ),
        ),
        # With `d_times_perm = np.inf`, currents should be produced
        # according to Ohm's law with `c` acting as conductance.
        (
            tf.constant(
                [
                    [
                        [
                            np.exp(-2 - np.log(1 / 1)),
                            np.exp(-2 - np.log(1 / 2)),
                            np.exp(-2 - np.log(1 / 3)),
                            np.exp(-2 - np.log(1 / 4)),
                        ],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [
                            -0.25 * np.exp(-2 - np.log(1 / 5)),
                            -0.25 * np.exp(-2 - np.log(1 / 6)),
                            -0.25 * np.exp(-2 - np.log(1 / 7)),
                            -0.25 * np.exp(-2 - np.log(1 / 8)),
                        ],
                    ],
                    [
                        [
                            np.exp(-2 - np.log(1 / 1)),
                            np.exp(-2 - np.log(1 / 2)),
                            np.exp(-2 - np.log(1 / 3)),
                            np.exp(-2 - np.log(1 / 4)),
                        ],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                ],
                dtype=tf.float32,
            ),
            tf.constant(
                [
                    [
                        np.exp(-2 - np.log(1 / 1)),
                        np.exp(-2 - np.log(1 / 2)),
                        np.exp(-2 - np.log(1 / 3)),
                        np.exp(-2 - np.log(1 / 4)),
                    ],
                    [
                        -0.25 * np.exp(-2 - np.log(1 / 5)),
                        -0.25 * np.exp(-2 - np.log(1 / 6)),
                        -0.25 * np.exp(-2 - np.log(1 / 7)),
                        -0.25 * np.exp(-2 - np.log(1 / 8)),
                    ],
                    [
                        np.exp(-2 - np.log(1 / 1)),
                        np.exp(-2 - np.log(1 / 2)),
                        np.exp(-2 - np.log(1 / 3)),
                        np.exp(-2 - np.log(1 / 4)),
                    ],
                ],
                dtype=tf.float32,
            ),
        ),
    ),
]


@pytest.mark.parametrize("args,expected", iv_nonlinearity_pf_I_testdata)
def test_iv_nonlinearity_pf_I(args, expected):
    I_ind_exp, I_exp = expected
    nonideality, V, G = args
    I, I_ind = nonideality.compute_I(V, G)
    utils.assert_tf_approx(I_ind, I_ind_exp)
    utils.assert_tf_approx(I, I_exp)


stuck_at_testdata = [
    (
        (
            nonidealities.StuckAt(2.0, 1.0),
            tf.constant(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
        ),
        tf.constant(
            [
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        ),
    ),
    (
        (
            nonidealities.StuckAt(5.0, 0.0),
            tf.constant(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
        ),
        tf.constant(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize("args,expected", stuck_at_testdata)
def test_stuck_at(args, expected):
    nonideality, G = args
    result = nonideality.disturb_G(G)
    utils.assert_tf_approx(result, expected)
