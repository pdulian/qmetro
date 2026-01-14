# QMetro++
## Python optimization package for large scale quantum metrology with customized strategy structures
QMetro++ is a Python package that
provides a set of tools for identifying optimal estimation protocols that
maximize quantum Fisher information (QFI). Optimization can be performed
for arbitrary configurations of input states, parameter-encoding channels,
noise correlations, control operations, and measurements. The use of tensor
networks and an iterative see-saw algorithm allows for an efficient
optimization even in the regime of a large number of channel uses.

Additionally, the package includes implementations of the recently
developed methods for computing fundamental upper bounds on QFI,
which serve as benchmarks for assessing the optimality of numerical
optimization results. All functionalities are wrapped up in a user-friendly
interface which enables the definition of strategies at various levels of
detail.

See detailed description in [our article](https://arxiv.org/abs/2506.16524) and [documentation](https://qmetro.readthedocs.io/en/latest/).

## Installation
To install QMetro++:

```
pip install qmetro
```

## Contact
For more information please contact: p.dulian@cent.uw.edu.pl
