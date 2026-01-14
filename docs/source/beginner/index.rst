.. _sec:basic:

Basic package usage---optimization of standard strategies
=========================================================

In this section we present the basic ways to use the package that
utilize high-level functions from :ref:`protocols <api:protocols>` and
:ref:`bounds <api:bounds>`
subpackages. These functions allow the user to compute and bound the
QFI within three standard scenarios: single-channel, parallel and
adaptive strategies, by simply specifying the parameter-encoding
channel and the number of uses.

In the following subsections we will apply these methods to study the
problem of phase estimation in the presence of two paradigmatic
decoherence models: dephasing (d) or amplitude damping (a).

The parameter-encoding channels :math:`\Lambda_\theta` that we will
consider are given by Kraus operators of the form

.. math::

   K_{\theta,k} = U_{\theta} K_k, \quad
   U_\theta= e^{-\frac{i}{2}\theta\sigma_z},

where :math:`\sigma_z` is a Pauli :math:`z`-matrix,
:math:`U_\theta` represents unitary parameter encoding, whereas
:math:`K_k` are Kraus operators corresponding to one of the two
decoherence models. For simplicity, we will always assume that
estimation is performed around the parameter value
:math:`\theta \approx 0`, so after differentiating the Kraus operators
we set :math:`\theta=0`---this does not affect the values of the QFI as
finite unitary rotations could always be incorporated in control
operations, measurements or the state preparation stage and hence will
not affect the optimized value of QFI.

In case of the *dephasing model*, the two Kraus operators may be
parametrized with a single parameter :math:`p\in[0, 1]`:

.. math::
   :label: eq:pardeph1

   K^{(d)}_0=\sqrt{p}\mathbb{1}, \quad
   K^{(d)}_1 = \sqrt{1-p}\sigma_z,

where :math:`p=1` corresponds to no dephasing, :math:`p=1/2` to the
strongest dephasing case, where all equatorial qubit states are mapped
to the maximally mixed state, while :math:`p=0` represents a phase flip
channel. Equivalently, we may use a different Kraus representation
where the two Kraus operators are given by:

.. math::
   :label: eq:pardeph2

   K^{(d)}_+=\frac{1}{\sqrt{2}} U_\epsilon, \quad
   K^{(d)}_-=\frac{1}{\sqrt{2}} U_{-\epsilon},

which may be interpreted as random rotations by angles
:math:`\pm \epsilon` with probability :math:`1/2` each---in order for
the two representations to match we need to fix
:math:`p=\cos^2(\epsilon/2)`.

In case of *amplitude* damping, the Kraus operators take the form

.. math::
   :label: eq:paramp

   K^{(a)}_0=\ket{0}\bra{0} +
   \sqrt{p} \ket{1}\bra{1},\quad
   K^{(a)}_1 = \sqrt{1-p} \ket{0}\bra{1},

where :math:`p=1` represents the no-decoherence case, while
:math:`p=0` the full relaxation case, where all the states of the qubit
are mapped to the ground :math:`\ket{0}` state.

Alternatively, dephasing and amplitude damping models can be considered
in a continuous time regime. In this case, the state undergoes a
process for some time :math:`t` which imprints on it information about
the parameter :math:`\omega` related to :math:`\theta` by
:math:`\theta=\omega t`. The derivative of the state with respect to
time is defined using the *Lindbladian operator*
:math:`\mathcal{L}_\omega`:

.. math::

   \frac{d\rho_{\omega, t}}{dt} =
   \mathcal{L}_{\omega}[\rho_{\omega, t}] =
   \frac{1}{i\hbar}[H_\omega, \rho_{\omega, t}] +
   \mathcal{D}[\rho_{\omega, t}],

where :math:`H_\omega = \hbar\omega\sigma_z/2` is a Hamiltonian
generating the evolution :math:`U_\theta` and :math:`\mathcal{D}` is a
dissipation term. The dissipation term has a general form:

.. math::

   \mathcal{D}[\rho] =
   \sum_i \gamma_i
   \left(
   L_i\rho L_i^\dagger -
   \frac{1}{2}\{L_i^\dagger L_i, \rho\}
   \right),

where :math:`\{\cdot,\cdot\}` is the anticommutator,
:math:`L_i` are jump operators that determine the type of dissipation
and :math:`\gamma_i\geq0` are damping rates. In case of dephasing and
amplitude damping only one pair :math:`L_i, \gamma_i` is required and
it is respectively:

.. math::

   L^{(d)} = \sigma_z/\sqrt{2}, \quad
   2p-1=e^{-\gamma^{(d)}t},

   L^{(a)} = \ket{0}\bra{1}, \quad
   p=e^{-\gamma^{(a)}t}.


.. toctree::
   :maxdepth: 1

   specchannel
   single
   parallel
   adaptive
   correlated
