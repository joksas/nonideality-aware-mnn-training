"""
Tests of functions of crossbar.nonlinear_IV
"""
import pytest
# pylint: disable=missing-function-docstring
import tensorflow as tf
from tests import utils

from crossbar import ideal

# In the ideal case, the bit-line outputs should represent the vector-matrix
# product of voltages and conductances.
compute_I_testdata = [
    (
        {
            "G": tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
            "V": tf.constant(
                [
                    [1.0, 0.0, -0.5],
                ]
            ),
        },
        tf.constant(
            [
                [
                    1.0 * 1.0 + 0.0 * 5.0 + (-0.5) * 9.0,
                    1.0 * 2.0 + 0.0 * 6.0 + (-0.5) * 10.0,
                    1.0 * 3.0 + 0.0 * 7.0 + (-0.5) * 11.0,
                    1.0 * 4.0 + 0.0 * 8.0 + (-0.5) * 12.0,
                ],
            ]
        ),
    ),
    (
        {
            "G": tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
            "V": tf.constant(
                [
                    [1.0, 0.0, -0.4],
                    [0.0, 2.0, 0.0],
                ]
            ),
        },
        tf.constant(
            [
                [
                    1.0 * 1.0 + 0.0 * 5.0 + (-0.4) * 9.0,
                    1.0 * 2.0 + 0.0 * 6.0 + (-0.4) * 10.0,
                    1.0 * 3.0 + 0.0 * 7.0 + (-0.4) * 11.0,
                    1.0 * 4.0 + 0.0 * 8.0 + (-0.4) * 12.0,
                ],
                [
                    0.0 * 1.0 + 2.0 * 5.0 + 0.0 * 9.0,
                    0.0 * 2.0 + 2.0 * 6.0 + 0.0 * 10.0,
                    0.0 * 3.0 + 2.0 * 7.0 + 0.0 * 11.0,
                    0.0 * 4.0 + 2.0 * 8.0 + 0.0 * 12.0,
                ],
            ]
        ),
    ),
]


@pytest.mark.parametrize("args,expected", compute_I_testdata)
def test_compute_I(args, expected):
    I = ideal.compute_I(**args)
    utils.assert_tf_approx(I, expected)


compute_I_all_testdata = [
    (
        {
            "G": tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
            "V": tf.constant(
                [
                    [0.2, -0.1, 0.0],
                ]
            ),
        },
        [
            tf.constant(
                [
                    [
                        0.2 * 1.0 + (-0.1) * 5.0 + 0.0 * 9.0,
                        0.2 * 2.0 + (-0.1) * 6.0 + 0.0 * 10.0,
                        0.2 * 3.0 + (-0.1) * 7.0 + 0.0 * 11.0,
                        0.2 * 4.0 + (-0.1) * 8.0 + 0.0 * 12.0,
                    ],
                ]
            ),
            tf.constant(
                [
                    [
                        [0.2 * 1.0, 0.2 * 2.0, 0.2 * 3.0, 0.2 * 4.0],
                        [(-0.1) * 5.0, (-0.1) * 6.0, (-0.1) * 7.0, (-0.1) * 8.0],
                        [0.0 * 9.0, 0.0 * 10.0, 0.0 * 11.0, 0.0 * 12.0],
                    ],
                ]
            ),
        ],
    ),
    (
        {
            "G": tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
            "V": tf.constant(
                [
                    [0.2, -0.1, 0.0],
                    [0.2, -0.1, 0.0],
                ]
            ),
        },
        [
            tf.constant(
                [
                    [
                        0.2 * 1.0 + (-0.1) * 5.0 + 0.0 * 9.0,
                        0.2 * 2.0 + (-0.1) * 6.0 + 0.0 * 10.0,
                        0.2 * 3.0 + (-0.1) * 7.0 + 0.0 * 11.0,
                        0.2 * 4.0 + (-0.1) * 8.0 + 0.0 * 12.0,
                    ],
                    [
                        0.2 * 1.0 + (-0.1) * 5.0 + 0.0 * 9.0,
                        0.2 * 2.0 + (-0.1) * 6.0 + 0.0 * 10.0,
                        0.2 * 3.0 + (-0.1) * 7.0 + 0.0 * 11.0,
                        0.2 * 4.0 + (-0.1) * 8.0 + 0.0 * 12.0,
                    ],
                ]
            ),
            tf.constant(
                [
                    [
                        [0.2 * 1.0, 0.2 * 2.0, 0.2 * 3.0, 0.2 * 4.0],
                        [(-0.1) * 5.0, (-0.1) * 6.0, (-0.1) * 7.0, (-0.1) * 8.0],
                        [0.0 * 9.0, 0.0 * 10.0, 0.0 * 11.0, 0.0 * 12.0],
                    ],
                    [
                        [0.2 * 1.0, 0.2 * 2.0, 0.2 * 3.0, 0.2 * 4.0],
                        [(-0.1) * 5.0, (-0.1) * 6.0, (-0.1) * 7.0, (-0.1) * 8.0],
                        [0.0 * 9.0, 0.0 * 10.0, 0.0 * 11.0, 0.0 * 12.0],
                    ],
                ]
            ),
        ],
    ),
]


@pytest.mark.parametrize("args,expected", compute_I_all_testdata)
def test_compute_I_all(args, expected):
    I_exp, I_ind_exp = expected
    I, I_ind = ideal.compute_I_all(**args)
    utils.assert_tf_approx(I, I_exp)
    utils.assert_tf_approx(I_ind, I_ind_exp)
