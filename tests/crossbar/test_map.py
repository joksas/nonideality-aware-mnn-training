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


# Test whether ideal crossbars compute vector-matrix products correctly.
ideal_dpe_testdata = [
        (
            {
                "x": tf.constant([
                    [1.0, -1.0],
                    [2.0, 0.0],
                    [0.0, -0.5],
                    ]),
                "w": tf.constant([
                    [1.0, -2.0, 3.0, -4.0],
                    [5.0, 6.0, -7.0, 8.0],
                    ]),
                },
            tf.constant([
                [
                    1.0*1.0 + (-1.0)*5.0,
                    1.0*(-2.0) + (-1.0)*6.0,
                    1.0*3.0 + (-1.0)*(-7.0),
                    1.0*(-4.0) + (-1.0)*8.0,
                    ],
                [
                    2.0*1.0 + 0.0*5.0,
                    2.0*(-2.0) + 0.0*6.0,
                    2.0*3.0 + 0.0*(-7.0),
                    2.0*(-4.0) + 0.0*8.0,
                    ],
                [
                    0.0*1.0 + (-0.5)*5.0,
                    0.0*(-2.0) + (-0.5)*6.0,
                    0.0*3.0 + (-0.5)*(-7.0),
                    0.0*(-4.0) + (-0.5)*8.0,
                    ],
                ]),
            )
        ]

@pytest.mark.parametrize("G_min,G_max", [
    (tf.constant(0.1), tf.constant(0.2)),
    (tf.constant(0.2), tf.constant(5.0)),
    ])
@pytest.mark.parametrize("V_ref", [
    tf.constant(0.2),
    tf.constant(1.0),
    ])
@pytest.mark.parametrize("is_ideal", [
    True,
    False,
    ])
@pytest.mark.parametrize("args,expected", ideal_dpe_testdata)
def test_ideal_dpe(args, expected, G_min, G_max, V_ref, is_ideal):
    x = args["x"]
    w = args["w"]

    k_V = 2*V_ref
    V = crossbar.map.x_to_V(x, k_V)

    G, max_weight = crossbar.map.w_to_G(w, G_min, G_max)

    if is_ideal:
        I = crossbar.ideal.compute_I(V, G)
    else:
        I, _ = crossbar.nonlinear_IV.compute_I_all(V, G, V_ref, tf.constant(2.0), n_std=tf.constant(0.0))

    y = crossbar.map.I_to_y(I, k_V, max_weight, G_max, G_min)

    utils.assert_tf_approx(y, expected)
