Quantum channel estimation problem
----------------------------------

In a paradigmatic quantum metrological problem, the goal is to estimate
the value of a single parameter :math:`\theta` encoded in some physical
process described by a quantum channel (formally a Completely Positive
Trace-Preserving map -- CPTP):

.. math::

   \Lambda_\theta: \mathcal{L}(\mathcal{I}) \mapsto \mathcal{L}(\mathcal{O}),

where :math:`\mathcal{I}`, :math:`\mathcal{O}` are Hilbert spaces of input
and output quantum systems and :math:`\mathcal{L}(\mathcal{H})` is a set
of linear operators on :math:`\mathcal{H}`, that is a superset of density
matrices on :math:`\mathcal{H}`.

The channel :math:`\Lambda_\theta` can be probed by an arbitrary
:math:`\theta`-independent input state

.. math::

   \rho_0 \in \mathcal{L}(\mathcal{I} \otimes \mathcal{A}),

where :math:`\mathcal{A}` is an auxiliary system called an *ancilla*;
:math:`\Lambda_\theta` does not act on :math:`\mathcal{A}`, but the
entanglement between :math:`\mathcal{I}` and :math:`\mathcal{A}` may
sometimes be used to enhance the estimation precision in the presence of
noise :cite:`Fujiwara2001, Kolodynski2013, kurdzialek2024, Liu2024`. The
resulting output state is

.. math::

   \rho_\theta := \left( \Lambda_\theta \otimes
   \mathbb{I}_{\mathcal{A}} \right)(\rho_0)
   \in \mathcal{L}(\mathcal{O} \otimes \mathcal{A}),

where :math:`\mathbb{I}_{\mathcal{A}}` represents the identity map on
:math:`\mathcal{L}(\mathcal{A})`---note the notational distinction from
:math:`\mathbb{1}_{\mathcal{A}}`, which denotes identity operator on the
:math:`\mathcal{A}` space.

The state :math:`\rho_\theta` is measured by a generalized measurement
:math:`\Pi=\{\Pi_i\}` resulting in the probability of outcome
:math:`i` equal to
:math:`p_\theta(i) := \mathrm{Tr}\!\left(\rho_\theta \Pi_i\right)`.

The parameter :math:`\theta` is then estimated using an estimator
:math:`\tilde\theta(i)` and the goal is to find the protocol producing
results as close as possible to the true value of :math:`\theta`. In
case of the simplest single-channel estimation strategy, as depicted in
:numref:`fig:intro` (A), the optimization of the protocol is equivalent
to optimizing the input state :math:`\rho_0` and the measurement
:math:`\{\Pi_i\}`. However, more complex protocols may additionally
involve optimization over control operations :math:`\textrm{C}_i`, as in
scenarios (C) and (D) in :numref:`fig:intro`.

In general, the exact form of the optimal protocol depends on the actual
estimation framework chosen, be it frequentist or Bayesian
:cite:`Hayashi2011, Meyer2023, DemkowiczDobrzanski2020`, as well as the
choice of the cost function/figure of merit to be minimized/maximized.
In this work, we adopt the frequentist approach, and focus on optimizing
the QFI as the figure of merit to be maximized.
