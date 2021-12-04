# Adjusting Training to Enable Accurate and Robust Low-Power Memristive Neural Networks

## Requirements

Python ≥3.9 and the packages listed in [requirements.txt](/requirements.txt).

## Repository structure

`awarememristor/crossbar`: memristor nonidealities and mapping onto crossbar arrays.

`awarememristor/training`: network training.

`awarememristor/simulations`: simulations presented in the manuscript.

`awarememristor/plotting`: figures presented in the manuscript.

## Reproducing results

To reproduce the simulations presented in the manuscript, execute
```text
python reproduce_paper.py
```

This might take a long time to finish, so you may want to split this file up in order to, for example, perform the simulations on multiple machines.

## Testing

To run unit tests, execute
```text
pytest tests
```
