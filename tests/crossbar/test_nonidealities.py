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


# I feel it is appropriate to use multiplication for expected tensors because
# it is not the underlying operation that we are testing. Writing it out
# reveals the logic behind the calculations that *should* take place - Dovydas
iv_nonlinearity_I_ind_testdata = [
    (
        (
            nonidealities.IVNonlinearity(1.0, 2.0, 1e-10),
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ),
            tf.constant(
                [
                    [1.0, 0.0],
                ]
            ),
        ),
        tf.constant(
            [
                [
                    [1.0 * 1.0, 1.0 * 2.0, 1.0 * 3.0, 1.0 * 4.0],
                    [0.0 * 5.0, 0.0 * 6.0, 0.0 * 7.0, 0.0 * 8.0],
                ],
            ]
        ),
    ),
    (
        (
            nonidealities.IVNonlinearity(2.0, 2.0, 1e-10),
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ),
            tf.constant(
                [
                    [1.0, 0.5],
                ]
            ),
        ),
        tf.constant(
            [
                [
                    [1.0 * 1.0, 1.0 * 2.0, 1.0 * 3.0, 1.0 * 4.0],
                    [0.5 * 5.0, 0.5 * 6.0, 0.5 * 7.0, 0.5 * 8.0],
                ],
            ]
        ),
    ),
    (
        (
            nonidealities.IVNonlinearity(0.5, 4.0, 1e-10),
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
            tf.constant(
                [
                    [0.0, 0.5, 1.0],
                ]
            ),
        ),
        tf.constant(
            [
                [
                    [0.0 * 1.0, 0.0 * 2.0, 0.0 * 3.0, 0.0 * 4.0],
                    # Baseline because V_ref = 0.5
                    [0.5 * 5.0, 0.5 * 6.0, 0.5 * 7.0, 0.5 * 8.0],
                    # Multiplying by additional factor of 4 because 1/0.5 = 2
                    # and n_avg = 4
                    [0.5 * 9.0 * 4, 0.5 * 10.0 * 4, 0.5 * 11.0 * 4, 0.5 * 12.0 * 4],
                ],
            ]
        ),
    ),
    (
        (
            nonidealities.IVNonlinearity(0.2, 3.0, 1e-10),
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ),
            tf.constant(
                [
                    [0.0, 0.2],
                    [0.1, 0.4],
                ]
            ),
        ),
        tf.constant(
            [
                [
                    [0.0 * 1.0, 0.0 * 2.0, 0.0 * 3.0, 0.0 * 4.0],
                    # Baseline because V_ref = 0.2
                    [0.2 * 5.0, 0.2 * 6.0, 0.2 * 7.0, 0.2 * 8.0],
                ],
                [
                    # Dividing by additional factor of 3 because 0.1/0.2 =
                    # 1/2 and n_avg = 3
                    [
                        0.2 * 1.0 / 3.0,
                        0.2 * 2.0 / 3.0,
                        0.2 * 3.0 / 3.0,
                        0.2 * 4.0 / 3.0,
                    ],
                    # Multiplying by additional factor of 3 because 0.4/0.2
                    # = 2 and n_avg = 3
                    [
                        0.2 * 5.0 * 3.0,
                        0.2 * 6.0 * 3.0,
                        0.2 * 7.0 * 3.0,
                        0.2 * 8.0 * 3.0,
                    ],
                ],
            ]
        ),
    ),
    (
        (
            nonidealities.IVNonlinearity(0.5, 5.0, 1e-10),
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ),
            tf.constant(
                [
                    [-0.5, -0.25, -1.0, 0.5],
                ]
            ),
        ),
        tf.constant(
            [
                [
                    # Baseline because V_ref = 0.5
                    [-0.5 * 1.0, -0.5 * 2.0, -0.5 * 3.0, -0.5 * 4.0],
                    # Dividing by additional factor of 5 because -0.25/-0.5
                    # = 1/2 and n_avg = 5
                    [
                        -0.5 * 5.0 / 5.0,
                        -0.5 * 6.0 / 5.0,
                        -0.5 * 7.0 / 5.0,
                        -0.5 * 8.0 / 5.0,
                    ],
                    # Multiplying by additional factor of 5 because
                    # -1.0/-0.5 = 1/2 and n_avg = 5
                    [
                        -0.5 * 9.0 * 5.0,
                        -0.5 * 10.0 * 5.0,
                        -0.5 * 11.0 * 5.0,
                        -0.5 * 12.0 * 5.0,
                    ],
                    # Baseline because V_ref = 0.5
                    [0.5 * 13.0, 0.5 * 14.0, 0.5 * 15.0, 0.5 * 16.0],
                ],
            ]
        ),
    ),
]


@pytest.mark.parametrize("args,expected", iv_nonlinearity_I_ind_testdata)
def test_iv_nonlinearity_I_ind(args, expected):
    nonideality, G, V = args
    _, result = nonideality.compute_I(V, G)
    utils.assert_tf_approx(result, expected)


iv_nonlinearity_I_testdata = [
    (
        (
            nonidealities.IVNonlinearity(5.0, 2.0, 1e-10),
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
            tf.constant(
                [
                    [1.0, 0.0, -0.5],
                    [0.0, 0.25, 0.0],
                ]
            ),
        ),
        [
            # With {n_avg = 2, n_std = 0} the bit-line outputs should
            # represent the vector-matrix product of voltages and
            # conductances.
            tf.constant(
                [
                    [
                        1.0 * 1.0 + 0.0 * 5.0 + (-0.5) * 9.0,
                        1.0 * 2.0 + 0.0 * 6.0 + (-0.5) * 10.0,
                        1.0 * 3.0 + 0.0 * 7.0 + (-0.5) * 11.0,
                        1.0 * 4.0 + 0.0 * 8.0 + (-0.5) * 12.0,
                    ],
                    [
                        0.0 * 1.0 + 0.25 * 5.0 + 0.0 * 9.0,
                        0.0 * 2.0 + 0.25 * 6.0 + 0.0 * 10.0,
                        0.0 * 3.0 + 0.25 * 7.0 + 0.0 * 11.0,
                        0.0 * 4.0 + 0.25 * 8.0 + 0.0 * 12.0,
                    ],
                ]
            ),
            # With {n_avg = 2, n_std = 0} currents should be produced
            # according to Ohm's law.
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
        ],
    ),
]


@pytest.mark.parametrize("args,expected", iv_nonlinearity_I_testdata)
def test_iv_nonlinearity_I(args, expected):
    I_exp, I_ind_exp = expected
    nonideality, G, V = args
    I, I_ind = nonideality.compute_I(V, G)
    utils.assert_tf_approx(I, I_exp)
    utils.assert_tf_approx(I_ind, I_ind_exp)


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
