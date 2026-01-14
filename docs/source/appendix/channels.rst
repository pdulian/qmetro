Quantum channel models
======================
QMetro++ package implements several commonly used quantum channels:

.. math::
    \Lambda_\theta(\rho) = \sum_k K_{\theta,k} \rho K_{\theta,k}^\dagger,

acting on qubit systems. All of them encode the parameter :math:`\theta`
via a unitary


.. math::
   :label: z-rotation

   U_\theta= e^{-\frac{i}{2}\theta\sigma_z},

where :math:`\sigma_z` is the Pauli z matrix and the parameter :math:`\theta`
is assumed to be in the neighborhood of :math:`0`.

The signal can be applied either before or after the noise action defined by
the Kraus operators :math:`\{K_k\}_k`:

- noise first: :math:`K_{\theta,k} = U_{\theta}K_k`.
- noise second: :math:`K_{\theta,k} = K_k U_{\theta}`,


.. _depolarization:

Depolarization
--------------

Depolarization is defined by the Kraus operators

.. math::

   K_0 = \sqrt{p}\, \mathbb{1},
   \quad
   K_i =\sqrt{\frac{1-p}{3}} \, \sigma_i \quad \mathrm{for} \quad i=x,y,z,

for some :math:`p \in [0, 1]`. This models the noise
:math:`\rho \rightarrow p\rho + \frac{1-p}{2}\mathbb{1}`.

Alternative Kraus representation is given by

.. math::

   K_0 = \sqrt{\frac{1+3\eta}{4}}\, \mathbb{1},
   \quad
   K_i =\sqrt{\frac{1-\eta}{4}} \, \sigma_i \quad \mathrm{for} \quad i=x,y,z,

for some :math:`\eta = (4p-1)/3 \in [-1/3, 1]`. Then the noise can be understood as
a uniform contraction of the Bloch ball by a factor :math:`\eta`.


.. _par-amp:

Parallel amplitude damping
--------------------------

Parallel amplitude damping channel is defined by the Kraus operators

.. math::

   K_0 = \begin{pmatrix}
   1 & 0 \\
   0 & \sqrt{1-p}
   \end{pmatrix},
   \quad
   K_1 = \begin{pmatrix}
   0 & \sqrt{p} \\
   0 & 0
   \end{pmatrix},

for some :math:`p \in [0,1]`. This models the decay
:math:`|1\rangle \rightarrow |0\rangle`.


.. _par-deph:

Parallel dephasing
-------------------------
Parallel dephasing channel is defined by the Kraus operators

.. math::

   K_0 = \sqrt{p}\, \mathbb{1},
   \quad
   K_1 = \sqrt{1-p} \, \sigma_z,

for some :math:`p \in [0,1]`. This models the dephasing noise that
shrinks the Bloch sphere along the x and y directions by a factor of 
:math:`2p-1`.

Alternative Kraus representation is given by rotations around the z-axis
(see :eq:`z-rotation`):

.. math::

    K_+=\frac{1}{\sqrt{2}} U_\epsilon,
    \quad
    K_-=\frac{1}{\sqrt{2}}  U_{-\epsilon},

where :math:`p = \cos^2(\epsilon/2)`.


.. _per-amp:

Perpendicular amplitude damping
-------------------------------

Perpendicular amplitude damping channel is defined by the Kraus operators

.. math::

   K_0 = | + \rangle \langle + | +\sqrt{p} | - \rangle \langle - |,
   \quad
   K_1 = \sqrt{1-p} | + \rangle \langle - |,

for some :math:`p \in [0,1]`. This models the decay
:math:`| - \rangle \rightarrow | + \rangle`.


.. _per-deph:

Perpendicular dephasing
-------------------------
Perpendicular dephasing channel is defined by the Kraus operators

.. math::

   K_0 = \sqrt{p}\, \mathbb{1},
   \quad
   K_1 = \sqrt{1-p} \, \sigma_x,

for some :math:`p \in [0,1]`. This models the dephasing noise that
shrinks the Bloch sphere along the y and z directions by a factor
:math:`2p-1`.
