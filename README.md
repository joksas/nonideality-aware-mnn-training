# Adjusting Training to Enable Accurate Low-Power Memristive Neural Networks

## Requirements

TensorFlow 2.0 or higher.

## Repository structure

`crossbar`: memristor nonidealities and mapping onto crossbar arrays.

`training`: network training.

`simulations`: simulations presented in the manuscript.

`plotting`: figures presented in the manuscript.

## Example

```python
from training.iterator import Iterator, Training, Inference, IVNonlinearity


DATASET = "mnist"
IDEAL = {
        "G_min": None,
        "G_max": None,
        "nonidealities": {}
        }
DEVICE_1 = {
        "G_min": 1/1003,
        "G_max": 1/284.6,
        "nonidealities": {"iv_nonlinearity": IVNonlinearity(2.132, 0.095)}
        }
DEVICE_2 = {
        "G_min": 1/1295000,
        "G_max": 1/366200,
        "nonidealities": {"iv_nonlinearity": IVNonlinearity(2.989, 0.369)}
        }

iterator = Iterator(
        DATASET,
        Training(num_repeats=2, num_epochs=100, batch_size=100, **IDEAL),
        [Inference(num_repeats=5, **DEVICE_1), Inference(num_repeats=3, **DEVICE_2)],
        )

iterator.train()
iterator.infer()
```
