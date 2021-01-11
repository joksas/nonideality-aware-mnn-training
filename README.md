# Non-idealility-aware Memristor-based NN Training

## Requirements

TensorFlow 2.0 or higher.

## Test

To train, go to `MNIST` and run `python Train.py`.

To train with baseline setup, go to `memristor_dense` under `memristor_utils.py` and change `def call()` accordingly (see comments inlined).

## Repo organisation

`model_architectures.py`: model topology.

`memristor_utils.py`: custom layers including memristor_dense.

`badmemristor`: copied over from Dovydas' repo.

`MNIST/Train.py`: training settings (lr, optimiser, epochs, batch size, etc).

## TODO

- [x] Add support for faulty devices.

- [ ] Experiment with lr-decay schedules.

- [x] Optimise forward propagation. Baseline takes 3s per epoch but ours takes 19s per epoch.
