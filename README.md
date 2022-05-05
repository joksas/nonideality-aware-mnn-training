# Nonideality-Aware Training for Accurate and Robust Low-Power Memristive Neural Networks

## Requirements

Python >=3.9 and the packages listed in [requirements.txt](/requirements.txt).

## Repository structure

`awarememristor/crossbar`: memristor nonidealities and mapping onto crossbar arrays.

`awarememristor/training`: network training.

`awarememristor/simulations`: simulations presented in the manuscript.

`awarememristor/plotting`: figures presented in the manuscript.

## Reproducing results

Script [reproduce_paper.py](/reproduce_paper.py) can be used to reproduce the simulations and plots presented in the manuscript.
Please follow the instructions in the script to obtain any missing experimental data (or comment out the function calls that require these data).
After that, execute
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

## Citation

A paper utilizing this package was published in [Advanced Science](https://onlinelibrary.wiley.com/journal/21983844) under the title ["Nonideality-Aware Training for Accurate and Robust Low-Power Memristive Neural Networks"](https://doi.org/10.1002/advs.202105784). To cite it, use the following BibTeX entry:

```bibtex
@article{JoksasWang2022,
  author       = {Joksas, Dovydas and Wang, Erwei and Barmpatsalos, Nikolaos and Ng, Wing H. and Kenyon, Anthony J. and Constantinides, George A. and Mehonic, Adnan},
  date         = {2022},
  doi          = {10.1002/advs.202105784},
  journaltitle = {Advanced Science},
  pages        = {2105784},
  title        = {Nonideality-Aware Training for Accurate and Robust Low-Power Memristive Neural Networks},
}
```
