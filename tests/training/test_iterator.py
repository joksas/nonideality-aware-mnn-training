"""
Tests of functions of training.iterator
"""
# pylint: disable=missing-function-docstring
import pytest

from awarememristor.crossbar import nonidealities
from awarememristor.training import iterator

nonideality_label_testdata = [
    (
        iterator.Stage(),
        "ideal",
    ),
    (
        iterator.Stage(
            nonidealities=[
                nonidealities.IVNonlinearityPF(0.5, (-1.0, -2.0, 1.5), (-5.0, -10.0, 0.0)),
            ],
        ),
        "IVNL_PF:-1_-2_1.5_-5_-10_0",
    ),
    (
        iterator.Stage(
            nonidealities=[nonidealities.StuckAt(1.20, 0.6341)],
        ),
        "Stuck:1.2_0.634",
    ),
    (
        iterator.Stage(
            nonidealities=[
                nonidealities.IVNonlinearityPF(0.5, (-1.0, -2.0, 1.5), (-5.0, -10.0, 0.0)),
                nonidealities.StuckAt(1.2344, 0.06341),
            ]
        ),
        "IVNL_PF:-1_-2_1.5_-5_-10_0+Stuck:1.23_0.0634",
    ),
]


@pytest.mark.parametrize("nonideal_instance,expected", nonideality_label_testdata)
def test_nonideality_label(nonideal_instance, expected):
    result = nonideal_instance.nonideality_label()
    assert result == expected


nonidealities_exception_testdata = [
    (
        [
            nonidealities.IVNonlinearityPF(0.5, (-1.0, -2.0, 1.5), (-5.0, -10.0, 0.0)),
            nonidealities.StuckAt(1.23, 0.0009),
            nonidealities.StuckAt(4.5, 0.1),
        ],
        "Current implementation does not support more than one linearity-preserving nonideality.",
    ),
    (
        [
            nonidealities.IVNonlinearityPF(0.5, (-1.0, -2.0, 1.5), (-5.0, -10.0, 0.0)),
            nonidealities.IVNonlinearityPF(0.5, (-4.0, -2.0, 1.5), (-7.0, -10.0, 0.0)),
        ],
        "Current implementation does not support more than one linearity-nonpreserving nonideality.",
    ),
]


@pytest.mark.parametrize("nonidealities_input,error_msg", nonidealities_exception_testdata)
def test_nonidealities_exception(nonidealities_input, error_msg):
    with pytest.raises(Exception) as exc:
        _ = iterator.Stage(nonidealities=nonidealities_input)
    assert error_msg in str(exc.value)
    assert exc.type == ValueError
