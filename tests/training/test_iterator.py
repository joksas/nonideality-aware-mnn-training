"""
Tests of functions of training.iterator
"""
# pylint: disable=missing-function-docstring
import pytest

from awarememristor.crossbar import nonidealities
from awarememristor.training import iterator

nonideality_label_testdata = [
    (
        iterator.Nonideal(),
        "ideal",
    ),
    (
        iterator.Nonideal(
            nonidealities=[nonidealities.IVNonlinearity(0.25, 1.53, 0.625)],
        ),
        "IVNL:1.53_0.625",
    ),
    (
        iterator.Nonideal(
            nonidealities=[nonidealities.StuckAt(1.20, 0.6341)],
        ),
        "Stuck:1.2_0.634",
    ),
    (
        iterator.Nonideal(
            nonidealities=[
                nonidealities.IVNonlinearity(0.25, 1.530, 0.123),
                nonidealities.StuckAt(1.2344, 0.06341),
            ]
        ),
        "IVNL:1.53_0.123+Stuck:1.23_0.0634",
    ),
]


@pytest.mark.parametrize("nonideal_instance,expected", nonideality_label_testdata)
def test_nonideality_label(nonideal_instance, expected):
    result = nonideal_instance.nonideality_label()
    assert result == expected


nonidealities_exception_testdata = [
    (
        [
            nonidealities.IVNonlinearity(0.25, 3.1, 0.1203),
            nonidealities.StuckAt(1.23, 0.0009),
            nonidealities.StuckAt(4.5, 0.1),
        ],
        "Current implementation does not support more than one linearity-preserving nonideality.",
    ),
    (
        [
            nonidealities.IVNonlinearity(0.25, 3.1, 0.1203),
            nonidealities.IVNonlinearity(0.25, 2.1, 0.1),
        ],
        "Current implementation does not support more than one linearity-nonpreserving nonideality.",
    ),
]


@pytest.mark.parametrize("nonidealities_input,error_msg", nonidealities_exception_testdata)
def test_nonidealities_exception(nonidealities_input, error_msg):
    with pytest.raises(Exception) as exc:
        _ = iterator.Nonideal(nonidealities=nonidealities_input)
    assert error_msg in str(exc.value)
    assert exc.type == ValueError
