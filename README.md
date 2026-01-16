# QMetro++
[![PyPI version](https://img.shields.io/pypi/v/qmetro)](https://pypi.org/project/qmetro/)
[![Documentation](https://img.shields.io/readthedocs/qmetro)](https://qmetro.readthedocs.io/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qmetro?period=total&units=INTERNATIONAL_SYSTEM&left_color=gray&right_color=blue&left_text=downloads)](https://pepy.tech/projects/qmetro)
[![License](https://img.shields.io/pypi/l/qmetro)](https://pypi.org/project/qmetro/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.16524-b31b1b.svg)](https://arxiv.org/abs/2506.16524)
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

First import may take a couple seconds (circa 1,86s) because QMetro++ loads
CVXPY and numerical backends.

## Contact
For more information please contact: p.dulian@cent.uw.edu.pl
