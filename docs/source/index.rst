.. qmetropp documentation master file, created by
   sphinx-quickstart on Tue Jun 17 16:28:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to QMetro++ documentation!
====================================

.. image:: https://img.shields.io/pypi/v/qmetro
   :target: https://pypi.org/project/qmetro/

.. image:: https://static.pepy.tech/personalized-badge/qmetro?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=blue&left_text=downloads
   :alt: Total PyPI downloads
   :target: https://pepy.tech/projects/qmetro

.. image:: https://img.shields.io/pypi/l/qmetro
   :target: https://pypi.org/project/qmetro/

.. image:: https://img.shields.io/badge/GitHub-qmetro-black?logo=github
   :alt: GitHub repository
   :target: https://github.com/pdulian/qmetro

.. image:: https://img.shields.io/badge/arXiv-2506.16524-b31b1b.svg
   :alt: arXiv paper
   :target: https://arxiv.org/abs/2506.16524


`QMetro++ <https://github.com/pdulian/qmetro>`_ is a Python package that
provides a set of tools for identifying optimal estimation protocols that
maximize quantum Fisher information (QFI). Optimization can be performed
for arbitrary configurations of input states, parameter-encoding channels,
noise correlations, control operations, and measurements. The use of tensor
networks and an iterative see-saw algorithm allows for an efficient
optimization even in the regime of a large number of channel uses
(:math:`N \approx 100`).

Additionally, the package includes implementations of the recently
developed methods for computing fundamental upper bounds on QFI,
which serve as benchmarks for assessing the optimality of numerical
optimization results. All functionalities are wrapped up in a user-friendly
interface which enables the definition of strategies at various levels of
detail.

See detailed description in `our article <https://arxiv.org/abs/2506.16524>`_.

Installation
-------------
To install QMetro++:

.. code-block:: console

   pip install qmetro

First import may take a couple seconds (circa 1,86s) because QMetro++ loads
CVXPY and numerical backends.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   theory/index
   beginner/index
   advanced/index
   appendix/index
   api/index
   bibliography

Contact
-------
For more information please contact: p.dulian@cent.uw.edu.pl
