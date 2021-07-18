# Adjusting Training to Enable Accurate Low-Power Memristive Neural Networks

## Requirements

TensorFlow 2.0 or higher.

## Test

To train, go to `MNIST`, set `path_to_project` in `MNIST/Train.py` and then run `python Train.py`.

## Repo organisation

`model_architectures.py`: model topology.

`memristor_utils.py`: custom layers including `memristor_dense`.

`crossbar`: mapping and nonidealities.

`MNIST/Train.py`: training setup.
