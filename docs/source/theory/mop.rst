.. _sec:mop:

Minimization over purifications
-------------------------------

The MOP method :cite:`Fujiwara2008, Escher2011, Demkowicz2012, kurdzialek2024`
is based on an observation that the QFI of a state
:math:`\rho_\theta\in\mathcal{L}(\mathcal{H})` is equal to the minimum of
QFI over its purifications:

.. math::

   F_Q(\rho_\theta) =
   \min_{\ket{\Psi_\theta}}
   F_Q(\ket{\Psi_\theta}\bra{\Psi_\theta}),

where :math:`\ket{\Psi_\theta} \in \mathcal{H} \otimes \mathcal{R}` is a
purification of :math:`\rho_\theta`,
:math:`\rho_\theta=\mathrm{Tr}_{\mathcal{R}}
\ket{\Psi_\theta}\bra{\Psi_\theta}`.

Consider now a parameter-encoding channel
:math:`\Lambda_\theta: \mathcal{L}(\mathcal{I}) \mapsto
\mathcal{L}(\mathcal{O})`. The idea of minimization over purifications
leads to an efficiently computable formula for the channel QFI
:cite:`Fujiwara2008, Escher2011, Demkowicz2012, kurdzialek2024`

.. math::
   :label: eq:minkraus

   F_Q(\Lambda_\theta) =
   4\min_{\{K_{\theta, k}\}}
   \left\|
   \sum_{k=1}^r
   \dot{K}_{\theta,k}^\dagger \dot{K}_{\theta,k}
   \right\|,

where :math:`\| \cdot \|` is the operator norm and the minimization is
performed over all equivalent Kraus representations of the channel,
such that:

.. math::

   \Lambda_\theta(\cdot)=
   \sum_{k=1}^r
   K_{\theta,k} \cdot K_{\theta,k}^\dagger.

This optimization becomes feasible, as one can effectively parametrize
the derivatives of the Kraus operators corresponding to all equivalent
Kraus representations as:

.. math::

   \dot K_{\theta, k}(h) =
   \dot{K}_{\theta, k}-
   i \sum_l h_{kl} K_{\theta, l},

where :math:`h \in \mathbb{C}^{ r\times r}` is a Hermitian matrix and
:math:`K_{\theta,k}` is any fixed Kraus representation of
:math:`\Lambda_\theta`, e.g. the canonical representation obtained from
eigenvectors of the corresponding Choi-Jamiołkowski (CJ) operator
:cite:`Bengtsson2006`, see Appendix :ref:`sec:link_prod` for the
definition of the CJ operator.

As a result, one may rephrase the apparently difficult minimization
problem :eq:`eq:minkraus` as:

.. math::

   F_Q(\Lambda_\theta) =
   4 \min_{h} \| \alpha(h)\|,

where

.. math::
   :label: eq:alpha

   \alpha(h) =
   \sum_{k}
   \dot K_{\theta, k}(h)^\dagger
   \dot K_{\theta, k}(h).

Importantly, this problem may be effectively written as a simple
semidefinite program (SDP)
:cite:`Demkowicz2012, kurdzialek2024`

.. math::
   :label: eq:mop_sdp

   F_Q(\Lambda_\theta) =
   4 \min_{\substack{\lambda \in \mathbb{R},\\ h=h^\dagger}}
   \lambda
   ~~\textrm{s.t.}~~ A \succeq 0,

where

.. math::

   A =
   \left(
   \begin{array}{c|ccc}
   \lambda \mathbb{1}_\mathcal{I} &
   \dot{{ K}}_{\theta,1}^\dagger(h) &
   ... &
   \dot{{ K}}_{\theta,r}^\dagger(h) \\ \hline
   \dot{{ K}}_{\theta,1}(h) & & & \\
   \vdots & & \mathbb{1}_{d \cdot r} & \\
   \dot{{ K}}_{\theta, r}(h) & & &
   \end{array}
   \right),

:math:`d= \textrm{dim} \mathcal{O}` and :math:`r` is the number of Kraus
operators. Provided the dimension of the relevant Hilbert space is small
enough, this problem may be solved efficiently using widely available
SDP solvers, such as :cite:`mosek`. This optimization is implemented in
the :py:func:`mop_channel_qfi
<qmetro.protocols.mop.mop_channel_qfi>` function.

.. _fig:linkadaptive:

.. figure:: /_static/img/adptive_rho.drawio.svg
   :width: 90%
   :align: center

   Final state resulting from the action of an adaptive protocol on
   :math:`N` parameter encoding channels, written via a formal link
   product operation between the corresponding CJ operators—quantum
   combs.

Notice that the dimension of ancilla :math:`\mathcal{A}` is not included
in any of the constraints in :eq:`eq:mop_sdp`. This is because MOP gives
the maximal QFI without any restriction on ancilla sizes. Furthermore,
in order to identify the optimal input state :math:`\ket\psi` one needs
to solve an additional SDP problem, see
:cite:`Zhou2021, kurdzialek2024`.

MOP can also be used to optimize QFI in case of multiple channel uses.
For the parallel strategy, :numref:`fig:intro` (B), one can simply apply
the single-channel method, replacing the channel :math:`\Lambda_\theta`
with :math:`\Lambda_\theta^{(N)}=\Lambda_\theta^{\otimes N}`—this is
implemented in the :py:func:`mop_parallel_qfi
<qmetro.protocols.mop.mop_parallel_qfi>` function.

In order to address an adaptive strategy, as in :numref:`fig:intro` (C),
one should first use the fact that a concatenated sequence of quantum
channels :math:`\mathrm{C}_i` where the free inputs and free output
spaces are respectively :math:`\mathcal{H}_{2i}`,
:math:`\mathcal{H}_{2i+1}`, :math:`i \in \{0, \dots, N-1\}`, can be
represented as a quantum comb :cite:`Chiribella2009`

.. math::

   C\in \mathrm{Comb}[
   (\mathcal{H}_0, \mathcal{H}_1), \dots,
   (\mathcal{H}_{2N-2}, \mathcal{H}_{2N-1})
   ],

which is a CJ operator satisfying:

.. math::
   :label: eq:comb_cond

   C \succeq 0, \quad C^{(N)}=C, \quad C^{(0)}=1, \\
   \mathrm{Tr}_{2k-1} C^{(k)} =
   \mathbb{1}_{2k-2} \otimes C^{(k-1)},
   \quad k\in \{1, \dots, N\}.

The above conditions guarantee that such quantum comb :math:`C` can
always be regarded as a concatenation of quantum channels
:math:`C_i`, namely :math:`C=C_0* \dots *C_{N-1}`, where :math:`*`
denotes the link product :cite:`Chiribella2009`, see Appendix
:ref:`sec:link_prod` for the definition of the link product.

In the context of metrological adaptive scenario, given a series of
channels :math:`\Lambda_{\theta, i}:
\mathcal{L}(\mathcal{I}_i)\mapsto \mathcal{L}(\mathcal{O}_i)`, we can
write the combined action of all the channels as:

.. math::
   :label: eq:uncorchan

   \Lambda^{(N)}_\theta =
   \Lambda_{\theta, 0} \otimes \dots \otimes
   \Lambda_{\theta, N-1}.

We may now represent the general adaptive strategy that includes the
input state as well as the control operations in the form of a quantum
comb

.. math::

   C \in \mathrm{Comb}[
   (\mathcal{\emptyset}, \mathcal{I}_0),
   (\mathcal{O}_0, \mathcal{I}_1), \dots,
   (\mathcal{O}_{N-3}, \mathcal{I}_{N-2}),
   (\mathcal{O}_{N-2},
   \mathcal{I}_{N-1}\otimes\mathcal{A})
   ]

and concatenate the operations using the link product in order to
obtain the output state

.. math::

   \rho_\theta =
   C * \mathit{\Lambda}_{\theta}^{(N)}
   \in
   \mathcal{L}(\mathcal{O}_{N-1}\otimes \mathcal{A}),

where :math:`\mathit{\Lambda}_{\theta}^{(N)}` (italics) formally
represents the CJ operator of :math:`\Lambda_{\theta}^{(N)}`, see
:numref:`fig:linkadaptive`. Here
:math:`\mathit{\Lambda}^{(N)}_\theta` does not necessarily need to
represent the action of uncorrelated channels as in
:eq:`eq:uncorchan`, but may also be a general parameter-dependent
quantum comb and in this way represent models with correlated noise or
non-local character of estimated parameter.

Optimization over adaptive protocols is thus equivalent to maximizing
the final QFI over quantum combs :math:`C`. Using again the MOP idea,
this problem may be effectively written down as an SDP (this time
slightly more involved) of the following form
:cite:`Altherr2021, Liu2023, kurdzialek2024`:

.. math::

   F_{\mathrm{Q}}^{(N)}\!\left(\Lambda_\theta\right) =
   4 \min_{\substack{\lambda \in \mathbb{R}, Q^{(k)}\\
   h=h^\dagger}}
   \lambda
   ~~\textrm{s.t.}~~ A \succeq 0,

   \underset{2\leq k \leq N-1}{\forall}
   \mathrm{Tr}_{\mathcal{O}_{k-1}} Q^{(k)} =
   \mathbb{1}_{\mathcal{I}_{k-1}} \otimes Q^{(k-1)},
   \quad Q^{(0)}= 1,

where
:math:`Q^{(k)} \in
\mathcal{L}(\mathcal{O}_0 \otimes \mathcal{I}_0 \otimes \dots \otimes
\mathcal{O}_{k-1} \otimes \mathcal{I}_{k-1})` are optimization
variables and :math:`A=` 

.. math::

   \left(
   \begin{array}{c|c}
   \mathbb{1}_{\mathcal{I}_{N-1}} \otimes Q^{(N-1)} &
   \ket{\dot{\tilde{K}}_{1,1}(h)} ...
   \ket{\dot{\tilde{K}}_{r,d}(h)} \\ \hline
   \bra{\dot{\tilde{K}}_{1,1}(h)} & \\
   \vdots & \lambda \mathbb{1}_{d r} \\
   \bra{\dot{\tilde{K}}_{r,d}(h)} &
   \end{array}
   \right)

with

.. math::

   \ket{\dot{\tilde{K}}_{i,k}(h)} =
   {_{\mathcal{O}_{N-1}}}\!\!
   \braket{i|\dot{K}_{\theta,k}^{(N)}(h)},

where

.. math::

   \ket{\dot{{K}}_{i,k}^{(N)}(h)} \in
   \mathcal{O}_0 \otimes \mathcal{I}_0 \otimes \dots \otimes
   \mathcal{O}_{N-1} \otimes \mathcal{I}_{N-1},

are derivatives of vectorized Kraus operators of
:math:`\Lambda_{\theta}^{(N)}`, :math:`r` is the number of Kraus
operators of :math:`\Lambda_\theta^{(N)}`,
:math:`d = \textrm{dim} \mathcal{O}_{N-1}`. This optimization is
implemented in the :py:func:`mop_adaptive_qfi
<qmetro.protocols.mop.mop_adaptive_qfi>` function.

Similarly to the single channel case, optimization of parallel and
adaptive strategies with MOP does not allow for the control of the
ancilla dimension and finding optimal input state or optimal comb
requires solving an additional SDP problem
:cite:`Liu2023, kurdzialek2024`. Moreover, since the algorithm requires
optimization over Kraus representations of
:math:`\Lambda^{(N)}_\theta` its time and memory complexity is
exponentially large in :math:`N`.
