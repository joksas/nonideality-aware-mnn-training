"""
Tests of functions of crossbar.nonlinear_IV
"""
# pylint: disable=missing-function-docstring
import tensorflow as tf
import pytest
import crossbar
from . import utils


w_params_to_G_testdata = [
        (
            {
                "weight_params": tf.constant([
                    [3.75, 2.5, 5.0, 2.5],
                    [2.5, 0.0, 0.0, 1.25],
                    ]),
                "max_weight": tf.constant(5.0),
                "G_min": tf.constant(2.0),
                "G_max": tf.constant(10.0),
                },
            tf.constant([
                [
                    [8.0, 6.0, 10.0, 6.0],
                    [6.0, 2.0, 2.0, 4.0],
                    ],
                ])
            ),
        (
            {
                "weight_params": tf.constant([
                    [8.0, -10.0],
                    [2.0, 0.0],
                    ]),
                "max_weight": tf.constant(8.0),
                "G_min": tf.constant(1.0),
                "G_max": tf.constant(3.0),
                },
            tf.constant([
                [
                    [3.0, 1.0],
                    [1.5, 1.0],
                    ],
                ])
            ),
        ]


@pytest.mark.parametrize("args,expected", w_params_to_G_testdata)
def test_w_params(args, expected):
    G = crossbar.map.w_params_to_G(**args)
    utils.assert_tf_approx(G, expected)
