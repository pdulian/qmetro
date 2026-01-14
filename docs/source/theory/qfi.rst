Quantum Fisher Information
--------------------------

Given some probabilistic model that specifies probabilities
:math:`p_\theta(i)` of obtaining different measurement outcomes
:math:`i` depending on the value of a parameter :math:`\theta`, one can
lower-bound the achievable mean squared error
:math:`\Delta^2 \tilde{\theta}` of any locally unbiased estimator
:math:`\tilde{\theta}(i)` via the famous *classical Cramér-Rao* (CCR)
bound :cite:`kay1993`:

.. math::
   :label: eq:cr_bound

   \Delta^2 \tilde\theta \ge \frac{1}{k F_C(p_{\theta})},

where :math:`k` is the number of independent repetitions of an
experiment and :math:`F_C` is *classical Fisher information* (CFI):

.. math::
   :label: eq:cfi

   F_C(p_{\theta}) := \sum_i
   \frac{\dot p_\theta(i)^2}{p_\theta(i)},

where :math:`\dot{p}_\theta` denotes the derivative with respect to
:math:`\theta`. The bound is asymptotically saturable, in the limit of
many repetitions of an experiment :math:`k \rightarrow \infty`, with
the help of e.g. a maximum-likelihood estimator :cite:`kay1993`. Hence,
the larger the CFI, the better parameter sensitivity of the model.

In quantum estimation models, the outcomes are the results of the
measurement :math:`\{\Pi_i\}` performed on a quantum state
:math:`\rho_\theta`,
:math:`p_\theta(i) = \mathrm{Tr}\!\left(\rho_\theta \Pi_i \right)`.
Hence, depending on the choice of the measurement, one may obtain larger
or smaller CFI. The QFI corresponds to the maximal achievable CFI, when
optimized over all admissible quantum measurements
:cite:`Helstrom1976, Braunstein1994, Paris2009`:

.. math::
   :label: eq:qfi

   F_Q(\rho_{\theta}) :=
   \max_{\{ \Pi_i \}_i} F_C(p_{\theta}) =
   \mathrm{Tr}\!\left(\rho_{\theta} L^2\right),

where :math:`L` is the *symmetric logarithmic derivative* (SLD) of
:math:`\rho_\theta`:

.. math::

   \dot \rho_\theta =
   \frac{1}{2}\left( \rho_{\theta}L + L\rho_{\theta} \right).

Moreover, the optimal measurement can, in particular, be chosen as the
projective measurement in the eigenbasis of the SLD.

Combining the above with the CCR bound :eq:`eq:cr_bound` leads to the
*quantum Cramér-Rao* (QCR) bound that sets a lower bound on achievable
estimation variance when estimating a parameter encoded in a quantum
state :math:`\rho_\theta`:

.. math::

   \Delta^2 \tilde{\theta} \geq
   \frac{1}{k F_Q(\rho_\theta)},

and is saturable in the limit of :math:`k \rightarrow \infty`.

This makes the QFI a fundamental concept in quantum estimation theory,
and makes it meaningful to formulate optimization problems in quantum
metrology in the form of an optimization of the QFI of the final quantum
state obtained at the output of a given metrological protocol.

The basic single-channel estimation task, depicted in
:numref:`fig:intro` (A), corresponds then to the following optimization
problem

.. math::
   :label: eq:channelqfi

   F_Q(\Lambda_\theta ) =
   \max_{\rho_0}
   F_Q\!\left[
   \Lambda_\theta \otimes \mathbb{I}_\mathcal{A}(\rho_0)
   \right].

We will refer to :math:`F_Q(\Lambda_\theta)` as the *channel* QFI, as it
corresponds to the maximal achievable QFI of the output state of the
channel, when the maximization over all input states is performed.

In case of protocols with multiple channel uses, :math:`N > 1`, one
could simply probe each channel independently, using independent probes
and independent measurements. In this case, by additivity of QFI, one
would end up with the corresponding :math:`N`-channels QFI ---
:math:`F_Q^{(N)}(\Lambda_\theta)` equal, for this simplistic strategy,
to :math:`N` times the single channel QFI.

However, one may also consider more sophisticated strategies involving
multiple channel uses,
which may provide significant advantage over the independent probing
strategy, in principle leading even to quadratic scaling of QFI ---
:math:`F_Q^{(N)}(\Lambda_\theta)\propto N^2`, referred to as the
*Heisenberg scaling* (HS) :cite:`Giovaennetti2006`.

In case of noiseless unitary estimation models
:math:`\Lambda_\theta(\rho) = U_\theta \rho U_\theta^\dagger` the
optimal probe states that maximize the QFI are easy to identify
analytically :cite:`Giovaennetti2006` and no sophisticated tools are
required. In case of noisy models, however, the actual optimization over
the protocol is much more involved, as in principle it requires
optimization over multipartite entangled states :numref:`fig:intro` (B)
or multiple control operations :math:`\textrm{C}_i` 
:numref:`fig:intro` (C) or both :numref:`fig:intro` (D).

This package allows for optimization of all kinds of metrological
strategies, employing ancillary systems, entanglement and control
operations, with the only physical restriction that the operations
applied have a definite causal order, and hence can be regarded as
so-called *quantum combs* in the sense formalized in
:cite:`Chiribella2009` and further in this text.

Regarding optimization strategies, we start with the discussion of
:ref:`MOP <sec:mop>` method, which is applicable to small-scale scenarios,
and then move on to discuss a more versatile :ref:`ISS <sec:iss>`
optimization which can be combined with
tensor-network formalism to deal with large-scale metrological problems.
