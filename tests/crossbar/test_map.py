import pytest
import tensorflow as tf

from awarememristor import crossbar
from tests import utils

double_w_to_G_testdata = [
    (
        {
            "double_w": tf.constant(
                [
                    [3.75, 2.5, 5.0, 2.5],
                    [2.5, 0.0, 0.0, 1.25],
                ]
            ),
            "G_off": tf.constant(2.0),
            "G_on": tf.constant(10.0),
        },
        [
            tf.constant(
                [
                    [8.0, 6.0, 10.0, 6.0],
                    [6.0, 2.0, 2.0, 4.0],
                ]
            ),
            tf.constant(5.0),
        ],
    ),
    (
        {
            "double_w": tf.constant(
                [
                    [8.0, 0.0],
                    [2.0, 0.0],
                ]
            ),
            "G_off": tf.constant(1.0),
            "G_on": tf.constant(3.0),
        },
        [
            tf.constant(
                [
                    [3.0, 1.0],
                    [1.5, 1.0],
                ]
            ),
            tf.constant(8.0),
        ],
    ),
]


@pytest.mark.parametrize("args,expected", double_w_to_G_testdata)
def test_double_w_to_G(args, expected):
    G_exp, max_weight_exp = expected
    G, max_weight = crossbar.map.double_w_to_G(**args)
    utils.assert_tf_approx(G, G_exp)
    utils.assert_tf_approx(max_weight, max_weight_exp)


w_to_G_testdata = [
    (
        {
            "weights": tf.constant(
                [
                    [3.75, 2.5, -5.0],
                    [-2.5, 0.0, 1.25],
                ]
            ),
            "G_off": tf.constant(2.0),
            "G_on": tf.constant(10.0),
        },
        [
            tf.constant(
                [
                    [8.0, 2.0, 6.0, 2.0, 2.0, 10.0],
                    [2.0, 6.0, 2.0, 2.0, 4.0, 2.0],
                ]
            ),
            tf.constant(5.0),
        ],
    ),
    (
        {
            "weights": tf.constant(
                [
                    [4.0],
                    [-2.0],
                ]
            ),
            "G_off": tf.constant(3.0),
            "G_on": tf.constant(5.0),
        },
        [
            tf.constant(
                [
                    [5.0, 3.0],
                    [3.0, 4.0],
                ]
            ),
            tf.constant(4.0),
        ],
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
            "x": tf.constant(
                [
                    [1.0, -1.0],
                    [2.0, 0.0],
                    [0.0, -0.5],
                ]
            ),
            "w": tf.constant(
                [
                    [1.0, -2.0, 3.0, -4.0],
                    [5.0, 6.0, -7.0, 8.0],
                ]
            ),
        },
        tf.constant(
            [
                [
                    1.0 * 1.0 + (-1.0) * 5.0,
                    1.0 * (-2.0) + (-1.0) * 6.0,
                    1.0 * 3.0 + (-1.0) * (-7.0),
                    1.0 * (-4.0) + (-1.0) * 8.0,
                ],
                [
                    2.0 * 1.0 + 0.0 * 5.0,
                    2.0 * (-2.0) + 0.0 * 6.0,
                    2.0 * 3.0 + 0.0 * (-7.0),
                    2.0 * (-4.0) + 0.0 * 8.0,
                ],
                [
                    0.0 * 1.0 + (-0.5) * 5.0,
                    0.0 * (-2.0) + (-0.5) * 6.0,
                    0.0 * 3.0 + (-0.5) * (-7.0),
                    0.0 * (-4.0) + (-0.5) * 8.0,
                ],
            ]
        ),
    )
]


@pytest.mark.parametrize(
    "G_off,G_on",
    [
        (tf.constant(0.1), tf.constant(0.2)),
        (tf.constant(0.2), tf.constant(5.0)),
    ],
)
@pytest.mark.parametrize(
    "k_V",
    [
        tf.constant(0.4),
        tf.constant(2.0),
    ],
)
@pytest.mark.parametrize("args,expected", ideal_dpe_testdata)
def test_ideal_dpe(args, expected, G_off, G_on, k_V):
    x = args["x"]
    w = args["w"]

    V = crossbar.map.x_to_V(x, k_V)

    G, max_weight = crossbar.map.w_to_G(w, G_off, G_on)

    I = crossbar.ideal.compute_I(V, G)

    y = crossbar.map.I_to_y(I, k_V, max_weight, G_on, G_off)

    utils.assert_tf_approx(y, expected)
