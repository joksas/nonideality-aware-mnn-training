import pytest
import tensorflow as tf

from awarememristor.crossbar import utils as crossbar_utils
from tests import utils

tf.random.set_seed(0)

random_bool_tensor_testdata = [
    (
        {
            "shape": [2, 3],
            "prob_true": 0.0,
        },
        tf.constant(
            [
                [False, False, False],
                [False, False, False],
            ]
        ),
    ),
    (
        {
            "shape": [4],
            "prob_true": 1.0,
        },
        tf.constant([True, True, True, True]),
    ),
]


@pytest.mark.parametrize("args,expected", random_bool_tensor_testdata)
def test_random_bool_tensor(args, expected):
    result = crossbar_utils.random_bool_tensor(**args)
    utils.assert_tf_bool_equal(result, expected)


multivariate_correlated_regression_testdata = [
    (
        (
            tf.constant(
                [1.0, 2.0, 3.0, 4.0],
            ),
            [2.0, 0.0],
            [1.0, -2.0],
            tf.constant(
                [
                    [1e-15, 0.0],
                    [0.0, 1e-15],
                ],
            ),
        ),
        tf.constant(
            [
                [3.0, 5.0, 7.0, 9.0],
                [-2.0, -2.0, -2.0, -2.0],
            ]
        ),
    ),
    (
        (
            tf.constant(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
            [2.0, 0.0],
            [1.0, -2.0],
            tf.constant(
                [
                    [1e-15, 0.0],
                    [0.0, 1e-15],
                ],
            ),
        ),
        tf.constant(
            [
                [
                    [3.0, 5.0, 7.0, 9.0],
                    [11.0, 13.0, 15.0, 17.0],
                    [19.0, 21.0, 23.0, 25.0],
                ],
                [
                    [-2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0],
                ],
            ]
        ),
    ),
]


@pytest.mark.parametrize("args,expected", multivariate_correlated_regression_testdata)
def test_multivariate_correlated_regression(args, expected):
    x, slopes, intercepts, cov_matrix = args
    result = crossbar_utils.multivariate_correlated_regression(x, slopes, intercepts, cov_matrix)
    utils.assert_tf_approx(result, expected)
