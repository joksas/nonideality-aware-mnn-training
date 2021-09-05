"""
Tests of functions of crossbar.nonlinear_IV
"""
# pylint: disable=missing-function-docstring
import tensorflow as tf
import pytest
from crossbar import nonlinear_IV
from . import utils


# I feel it is appropriate to use multiplication for expected tensors because
# it is not the underlying operation that we are testing. Writing it out
# reveals the logic behind the calculations that *should* take place - Dovydas
test_compute_currents_testdata = [
        (
            {
                "n_avg": tf.constant(2.0),
                "n_std": tf.constant(0.0),
                "V_ref": tf.constant(0.5),
                "G": tf.constant([
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    ]),
                "V": tf.constant([
                    [1.0, 0.0],
                    ]),
                },
            tf.constant([
                [
                    [1.0*1.0, 1.0*2.0, 1.0*3.0, 1.0*4.0],
                    [0.0*5.0, 0.0*6.0, 0.0*7.0, 0.0*8.0],
                    ],
                ])
            ),
        (
            {
                "n_avg": tf.constant(2.0),
                "n_std": tf.constant(0.0),
                "V_ref": tf.constant(0.5),
                "G": tf.constant([
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    ]),
                "V": tf.constant([
                    [1.0, 0.5],
                    ]),
                },
            tf.constant([
                [
                    [1.0*1.0, 1.0*2.0, 1.0*3.0, 1.0*4.0],
                    [0.5*5.0, 0.5*6.0, 0.5*7.0, 0.5*8.0],
                    ],
                ])
            ),
        (
            {
                "n_avg": tf.constant(4.0),
                "n_std": tf.constant(0.0),
                "V_ref": tf.constant(0.5),
                "G": tf.constant([
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    ]),
                "V": tf.constant([
                    [0.0, 0.5, 1.0],
                    ]),
                },
            tf.constant([
                [
                    [0.0*1.0, 0.0*2.0, 0.0*3.0, 0.0*4.0],
                    # Baseline because V_ref = 0.5
                    [0.5*5.0, 0.5*6.0, 0.5*7.0, 0.5*8.0],
                    # Multiplying by additional factor of 4 because 1/0.5 = 2
                    # and n_avg = 4
                    [0.5*9.0*4, 0.5*10.0*4, 0.5*11.0*4, 0.5*12.0*4],
                    ],
                ])
            ),
        (
                {
                    "n_avg": tf.constant(3.0),
                    "n_std": tf.constant(0.0),
                    "V_ref": tf.constant(0.2),
                    "G": tf.constant([
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        ]),
                    "V": tf.constant([
                        [0.0, 0.2],
                        [0.1, 0.4],
                        ]),
                    },
                tf.constant([
                    [
                        [0.0*1.0, 0.0*2.0, 0.0*3.0, 0.0*4.0],
                        # Baseline because V_ref = 0.2
                        [0.2*5.0, 0.2*6.0, 0.2*7.0, 0.2*8.0],
                        ],
                    [
                        # Dividing by additional factor of 3 because 0.1/0.2 =
                        # 1/2 and n_avg = 3
                        [0.2*1.0/3.0, 0.2*2.0/3.0, 0.2*3.0/3.0, 0.2*4.0/3.0],
                        # Multiplying by additional factor of 3 because 0.4/0.2
                        # = 2 and n_avg = 3
                        [0.2*5.0*3.0, 0.2*6.0*3.0, 0.2*7.0*3.0, 0.2*8.0*3.0],
                        ],
                    ])
                ),
        (
                {
                    "n_avg": tf.constant(5.0),
                    "n_std": tf.constant(0.0),
                    "V_ref": tf.constant(0.5),
                    "G": tf.constant([
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0],
                        ]),
                    "V": tf.constant([
                        [-0.5, -0.25, -1.0, 0.5],
                        ]),
                    },
                tf.constant([
                    [
                        # Baseline because V_ref = 0.5
                        [-0.5*1.0, -0.5*2.0, -0.5*3.0, -0.5*4.0],
                        # Dividing by additional factor of 5 because -0.25/-0.5
                        # = 1/2 and n_avg = 5
                        [-0.5*5.0/5.0, -0.5*6.0/5.0, -0.5*7.0/5.0, -0.5*8.0/5.0],
                        # Multiplying by additional factor of 5 because
                        # -1.0/-0.5 = 1/2 and n_avg = 5
                        [-0.5*9.0*5.0, -0.5*10.0*5.0, -0.5*11.0*5.0, -0.5*12.0*5.0],
                        # Baseline because V_ref = 0.5
                        [0.5*13.0, 0.5*14.0, 0.5*15.0, 0.5*16.0],
                        ],
                    ])
                ),
        ]


@pytest.mark.parametrize("args,expected", test_compute_currents_testdata)
def test_compute_currents(args, expected):
    I = nonlinear_IV.compute_currents(**args)
    utils.assert_tf_approx(I, expected)
