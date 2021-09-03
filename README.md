# Adjusting Training to Enable Accurate Low-Power Memristive Neural Networks

## Requirements

TensorFlow 2.0 or higher.

## Repository structure

`crossbar`: memristor nonidealities and mapping onto crossbar arrays.

`training`: network training.

## Example

```python
from training.iterator import Iterator, Training, Inference, IVNonlinearity


dataset = "MNIST"
G_off = 1/983.3
G_on = 1/281.3
iv_nonlinearity = IVNonlinearity("low-resistance", 2.132, 0.095)

iterator = Iterator(
	dataset,
	G_off,
	G_on,
        Training(num_repeats=2, num_epochs=100, batch_size=100, iv_nonlinearity=iv_nonlinearity),
        Inference(num_repeats=3, iv_nonlinearity=iv_nonlinearity),
        )

iterator.train()
iterator.infer()
```
