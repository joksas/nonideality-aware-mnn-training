# Nonideality-Aware Training for Accurate and Robust Low-Power Memristive Neural Networks

## Requirements

Python ≥3.9 and the packages listed in [requirements.txt](/requirements.txt).

## Repository structure

`awarememristor/crossbar`: memristor nonidealities and mapping onto crossbar arrays.

`awarememristor/training`: network training.

`awarememristor/simulations`: simulations presented in the manuscript.

`awarememristor/plotting`: figures presented in the manuscript.

## Reproducing results

To reproduce the simulations and plots presented in the manuscript, execute
```text
python reproduce_paper.py
```

This might take a long time to finish, so you may want to split this file up in order to, for example, perform the simulations on multiple machines.

## Testing

To run unit tests, execute
```text
pytest tests
```

## Using this package

**This package should not be used in production.**
The code is extensible but was written mostly with specific [simulations](/awarememristor/simulations) in mind.
Any new functionality (such as different nonidealities) should be incorporated carefully.
For example, to handle combinations of *multiple* linearity-preserving nonidealities (which is not currently supported), it may *not* be sufficient to simply apply them one after another.
