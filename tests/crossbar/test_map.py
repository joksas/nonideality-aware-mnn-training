"""
Tests of functions of crossbar.nonlinear_IV
"""
# pylint: disable=missing-function-docstring
import tensorflow as tf
import pytest
import crossbar
from tests import utils


w_params_to_G_testdata = [
        (
            {
                "weight_params": tf.constant([
                    [3.75, 2.5, 5.0, 2.5],
                    [2.5, 0.0, 0.0, 1.25],
                    ]),
                "G_min": tf.constant(2.0),
                "G_max": tf.constant(10.0),
                },
            [
                tf.constant([
                    [8.0, 6.0, 10.0, 6.0],
                    [6.0, 2.0, 2.0, 4.0],
                    ]),
                tf.constant(5.0),
                ]
            ),
        (
            {
                "weight_params": tf.constant([
                    [8.0, -10.0],
                    [2.0, 0.0],
                    ]),
                "G_min": tf.constant(1.0),
                "G_max": tf.constant(3.0),
                },
            [
                tf.constant([
                    [3.0, 1.0],
                    [1.5, 1.0],
                    ]),
                tf.constant(8.0),
                ]
            ),
        ]


@pytest.mark.parametrize("args,expected", w_params_to_G_testdata)
def test_w_params_to_G(args, expected):
    G_exp, max_weight_exp = expected
    G, max_weight = crossbar.map.w_params_to_G(**args)
    utils.assert_tf_approx(G, G_exp)
    utils.assert_tf_approx(max_weight, max_weight_exp)


w_to_G_testdata = [
        (
            {
                "weights": tf.constant([
                    [3.75, 2.5, -5.0],
                    [-2.5, 0.0, 1.25],
                    ]),
                "G_min": tf.constant(2.0),
                "G_max": tf.constant(10.0),
                },
            [
                tf.constant([
                    [8.0, 2.0, 6.0, 2.0, 2.0, 10.0],
                    [2.0, 6.0, 2.0, 2.0, 4.0, 2.0],
                    ]),
                tf.constant(5.0),
                ]
            ),
        (
            {
                "weights": tf.constant([
                    [4.0],
                    [-2.0],
                    ]),
                "G_min": tf.constant(3.0),
                "G_max": tf.constant(5.0),
                },
            [
                tf.constant([
                    [5.0, 3.0],
                    [3.0, 4.0],
                    ]),
                tf.constant(4.0),
                ]
            ),
        ]


@pytest.mark.parametrize("args,expected", w_to_G_testdata)
def test_w_to_G(args, expected):
    G_exp, max_weight_exp = expected
    G, max_weight = crossbar.map.w_to_G(**args)
    utils.assert_tf_approx(G, G_exp)
    utils.assert_tf_approx(max_weight, max_weight_exp)
