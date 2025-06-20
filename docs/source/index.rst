.. qmetropp documentation master file, created by
   sphinx-quickstart on Tue Jun 17 16:28:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to QMetro++ documentation!
====================================
`QMetro++ <https://github.com/pdulian/qmetro>`_ is a Python package
containing a set of tools dedicated to
identifying optimal estimation protocols that maximize quantum Fisher
information (QFI). Optimization can be performed for an arbitrary
arrangement of input states, parameter encoding channels, noise
correlations, control operations and measurements. The use of tensor
networks and iterative see-saw algorithm allows for an efficient
optimization even in the regime of large number of channel uses
(:math:`N \approx 100`).

Additionally, the package comes with an implementation of the recently
developed methods for computing fundamental upper bounds on QFI, which
serve as benchmarks of optimality of the outcomes of numerical
optimization. All functionalities are wrapped up in a user-friendly
interface which enables defining strategies at various levels of detail.

See detiled description in `our article <>`_.

Installation
=============
QMetro++ requires `ncon <https://github.com/mhauru/ncon>`_ package for tensor network contraction. To install ncon:

.. code-block:: console
   
   pip install --user 

Then to install QMetro++:

.. code-block:: console

   pip install --user qmetropp


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   qmetro

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
