import pytest
import tensorflow as tf

from awarememristor.training import utils
from tests import utils as test_utils

compute_avg_crossbar_power_testdata = [
    (
        tf.constant(
            [
                [
                    1.0,
                    0.0,
                    2.0,
                ],
            ]
        ),
        tf.constant(
            [
                [
                    [0.0, 1.0],
                    [2.0, 3.0],
                    [4.0, 5.0],
                ],
            ]
        ),
        19.0,
    ),
    (
        tf.constant(
            [
                [
                    4.0,
                    1.0,
                ],
                [
                    0.0,
                    1.0,
                ],
            ]
        ),
        tf.constant(
            [
                [
                    [0.0, 1.0],
                    [2.0, 3.0],
                ],
                [
                    [2.0, 4.0],
                    [0.0, 2.0],
                ],
            ],
        ),
        5.5,
    ),
    (
        tf.constant(
            [
                [
                    -1.0,
                    0.0,
                    2.0,
                ],
            ]
        ),
        tf.constant(
            [
                [
                    [0.0, 1.0],
                    [2.0, 3.0],
                    [-4.0, 5.0],
                ],
            ]
        ),
        19.0,
    ),
    (
        tf.constant(
            [
                [
                    -4.0,
                    1.0,
                ],
                [
                    0.0,
                    -1.0,
                ],
            ]
        ),
        tf.constant(
            [
                [
                    [0.0, -1.0],
                    [2.0, -3.0],
                ],
                [
                    [2.0, 4.0],
                    [0.0, -2.0],
                ],
            ],
        ),
        5.5,
    ),
]


@pytest.mark.parametrize("V,I_ind,expected", compute_avg_crossbar_power_testdata)
def test_compute_avg_crossbar_power(V, I_ind, expected):
    result = utils.compute_avg_crossbar_power(V, I_ind)
    assert pytest.approx(result) == expected
