.. _sec:specchannel:

Specifying the parameter-dependent channel
------------------------------------------

First, let us start by explaining how to encode the channel using the
:py:class:`ParamChannel <qmetro.param_channel.param_channel.ParamChannel>`
class. We can do that in one of the four available ways:

- from a list of Kraus operators and their derivatives with respect to
  :math:`\theta`,
- from a CJ matrix and its derivative over :math:`\theta`,
- using a predefined creator function,
- from a Lindbladian, its derivative with respect to :math:`\omega` and
  a specified evolution time.

The following listing shows how the above options are implemented in
practice using the example of the dephasing channel
:eq:`eq:pardeph1`:

.. code-block:: python

   import numpy as np
   from qmetro import *

   # Pauli z-matrix:
   sz = np.array([
       [1, 0],
       [0, -1]
   ])

   # Kraus operators and their derivatives for dephasing channel:
   p = 0.75
   krauses = [np.sqrt(p) * np.identity(2), np.sqrt(1-p) * sz]
   dkrauses = [-1j/2 * sz @ K for K in krauses]

   # ParamChannel instance created from Kraus operators:
   channel1 = ParamChannel(krauses=krauses, dkrauses=dkrauses)

   # CJ matrix and its derivative created from Kraus operators:
   choi = choi_from_krauses(krauses)
   dchoi = dchoi_from_krauses(krauses, dkrauses)

   # ParamChannel instance created from CJ matrix:
   channel2 = ParamChannel(choi=choi, dchoi=dchoi)

   # ParamChannel instance created using creator function:
   channel3 = par_dephasing(p)

By design the :py:class:`ParamChannel <qmetro.param_channel.param_channel.ParamChannel>`
class represents a discrete time quantum evolution. Thus to create its
objects from :math:`\mathcal{L}_\omega, \dot{\mathcal{L}}_\omega` one
needs to move from continuous to discrete time regime by integrating
over specified evolution time :math:`t`. This can be done using
:py:func:`choi_from_lindblad <qmetro.qtools.choi_from_lindblad>` function
from :mod:`qtools <qmetro.qtools>` which performs this integration for 
time-independent Lindbladian specified either as a function or
a pair of Hamiltonian and a list of jump operators:

.. code-block:: python

   # continuing previous example

   # Lindbladian and its derivative.
   omega = 0.0  # Parameter.
   t = 1        # Time.
   # Dephasing strength:
   gamma = -np.log(2*p-1)/t
   # Hilbert space dimension
   dim = 2

   def lindblad(rho):
       # Rotation around z part:
       rot = 0.5j*omega * (rho@sz - sz@rho)
       # Dephasing part:
       deph = gamma * (sz@rho@sz - rho)
       return rot + deph

   def dlindblad(rho):
       return 0.5j * (rho@sz - sz@rho)

   choi, dchoi = choi_from_lindblad(lindblad, dlindblad, t, dim=dim)
   channel5 = ParamChannel(choi=choi, dchoi=dchoi)

   # or

   # Hamiltonian
   H = 0.5 * omega * sz
   # Jump operators rescaled by sqrt(gamma)
   Ls = [np.sqrt(gamma/2) * sz]
   # Derivative of the Hamiltonian
   dH = 0.5 * sz
   # Derivative of the jump operators
   dLs = [np.zeros_like(L) for L in Ls]
   choi, dchoi = choi_from_lindblad((H, Ls), (dH, dLs), t)
   channel6 = ParamChannel(choi=choi, dchoi=dchoi)

Objects of the :py:class:`ParamChannel <qmetro.param_channel.param_channel.ParamChannel>`
class can be used to compute
:math:`\rho_\theta = \Lambda_\theta(\rho)` and
:math:`\dot\rho_\theta = \frac{d}{d\theta}\Lambda_\theta(\rho)`, e.g.:

.. code-block:: python

   channel = channel1
   rho = np.array([
       [1, 0],
       [0, 0]
   ])

   rho_t, drho_t = channel(rho)

or to obtain the CJ operator from Kraus operators and vice versa, e.g.:

.. code-block:: python

   channel1 = ParamChannel(krauses=krauses, dkrauses=dkrauses)
   # CJ matrix:
   choi = channel.choi()
   # Derivative of CJ matrix:
   dchoi = channel.dchoi()

   channel2 = ParamChannel(choi=choi, dchoi=dchoi)
   # Kraus operators:
   krauses = channel2.krauses()
   # Kraus operators and their derivatives:
   krauses, dkrauses = channel2.dkrauses()

They can be combined with each other to create new channels using:

- scalar multiplication
  :math:`\left(\alpha \Lambda_\theta \right)(\rho) =
  \alpha\Lambda_\theta(\rho)`, e.g.:

  .. code-block:: python

     a = 0.2
     new_channel = channel.scalar_mul(a)
     # or equivalently
     new_channel = a * channel

- addition
  :math:`(\Lambda_\theta + \mathrm{\Phi}_\theta)(\rho) =
  \Lambda_\theta(\rho) + \Phi_\theta(\rho)`, e.g.:

  .. code-block:: python

     new_channel = channel1.add(channel2)
     # or equivalently
     new_channel = channel1 + channel2

- composition
  :math:`(\Lambda_\theta \circ \mathrm{\Phi}_\theta)(\rho) =
  \Lambda_\theta\left( \Phi_\theta(\rho) \right)`, e.g.:

  .. code-block:: python

     new_channel = channel1.compose(channel2)
     # or equivalently
     new_channel = channel1 @ channel2

- Kronecker product
  :math:`\left( \Lambda_\theta \otimes \Phi_\theta \right)
  (\rho_1 \otimes \rho_2) =
  \Lambda_\theta(\rho_1) \otimes \Phi_\theta(\rho_2)`, e.g.:

  .. code-block:: python

     new_channel = channel1.kron(channel2)

- Kronecker power
  :math:`\Lambda_\theta^{\otimes N} =
  \underbrace{\Lambda_\theta \otimes \dots \otimes \Lambda_\theta}_{N}`,
  e.g.:

  .. code-block:: python

     N = 3
     new_channel = channel1.kron_pow(N)
