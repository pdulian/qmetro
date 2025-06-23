# QMetro++
## Python optimization package for large scale quantum metrology with customized strategy structures
QMetro++ is a Python package containing a set of tools dedicated to
identifying optimal estimation protocols that maximize quantum Fisher
information (QFI). Optimization can be performed for an arbitrary
arrangement of input states, parameter encoding channels, noise
correlations, control operations and measurements. The use of tensor
networks and iterative see-saw algorithm allows for an efficient
optimization even in the regime of large number of channel uses.

Additionally, the package comes with an implementation of the recently
developed methods for computing fundamental upper bounds on QFI, which
serve as benchmarks of optimality of the outcomes of numerical
optimization. All functionalities are wrapped up in a user-friendly
interface which enables defining strategies at various levels of detail.

See detailed description in [our article]() and [documentation](https://qmetro.readthedocs.io/en/latest/).

## Installation
QMetro++ requires [ncon](https://github.com/mhauru/ncon) package for tensor network contraction. To install ncon:
```
pip install --user ncon
```
Then to install QMetro++:

```
pip install qmetro
```

## Contact
For more information please contact: p.dulian@cent.uw.edu.pl