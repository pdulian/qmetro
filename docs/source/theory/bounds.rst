.. _sec:bounds:

Upper bounds
------------

In Sec. :ref:`sec:mop` we explained that the QFI can be computed via
minimization of the norm of :math:`\alpha` :eq:`eq:alpha` over all Kraus
representations :math:`\{K_{\theta, k}\}` of the channel
:math:`\Lambda_\theta`. It follows that in order to compute QFI for
parallel or adaptive strategy we need to optimize over Kraus
representations :math:`\{K^{(N)}_{\theta, k}\}` of the whole channel
:math:`\Lambda^{(N)}_\theta=\Lambda^{\otimes N}_\theta`—a task that is
exponentially hard in :math:`N`. However, if we only want to obtain the
upper bounds, we may dramatically reduce the complexity of the
optimization, and reformulate the computation of the bound in a way
that only minimization over Kraus representations of the *single
channel* :math:`\Lambda_\theta` is required
:cite:`Demkowicz2012, Kurdzialek2023, kurdzialek2024bounds, Demkowicz2014, Kolodynski2013`.

In case of parallel strategies this leads to an upper bound of the form
:cite:`Escher2011, Demkowicz2012, Kolodynski2013`:

.. math::
   :label: eq:bound_par

   F^{(N)}_Q(\Lambda_\theta) \le
   4N \min_{h}
   \left(
   \|\alpha(h)\| + (N-1)\|\beta(h)\|^2
   \right),

where :math:`\alpha(h)` is defined in :eq:`eq:alpha`, while

.. math::

   \beta(h) =
   \sum_k
   \dot K_{\theta, k}^\dagger(h) K_{\theta, k}(h).

This bound allows immediate exclusion of the possibility of the
Heisenberg scaling in the model—it is enough to show that there exist
:math:`h` for which :math:`\beta(h)=0`, which may be formulated as a
simple algebraic condition :cite:`Demkowicz2012, Zhou2021`.

In case of the adaptive strategy, one may again employ the idea of MOP
to bound the maximal increase of QFI when the adaptive strategy is
extended by one additional step including one more sensing channel
:cite:`Kurdzialek2023`:

.. math::

   F_{\mathrm{Q}}^{(N+1)}(\Lambda_\theta) \le
   F^{(N)}_{\mathrm{Q}}(\Lambda_\theta) +
   4\min_{h}
   \left[
   \|\alpha(h)\| +
   \sqrt{F^{(N)}_{\mathrm{Q}}(\Lambda_\theta)} \|\beta(h)\|
   \right].

Both of these bounds can be computed efficiently regardless of the
value of :math:`N` :cite:`Kurdzialek2023`. They are implemented in the
:py:func:`par_bounds <qmetro.bounds.bounds.par_bounds>` and
:py:func:`ad_bounds <qmetro.bounds.bounds.ad_bounds>` functions respectively.

Importantly, the bounds for parallel and adaptive strategies are
asymptotically (for :math:`N\rightarrow\infty`) equivalent and
saturable :cite:`Zhou2021, Kurdzialek2023`—there are strategies that
asymptotically achieve the optimal value of QFI predicted by the
bounds.

Depending on whether there exist :math:`h` for which
:math:`\beta(h)=0` we may have two possible asymptotic behaviors,
namely the *standard scaling* (SS):

.. math::
   :label: eq:asym_ss

   \lim_{N\rightarrow\infty}
   F^{(N)}_{Q}(\Lambda_\theta)/N =
   4\min_{h,\beta(h)=0}\|\alpha(h)\|,

and the *Heisenberg scaling* (HS):

.. math::
   :label: eq:asym_hs

   \lim_{N\rightarrow\infty}
   F^{(N)}_{Q}(\Lambda_\theta)/N^2 =
   4\min_h\|\beta(h)\|^2.

In the package, the asymptotic bound can be computed using
:py:func:`asym_scaling_qfi <qmetro.bounds.bounds.asym_scaling_qfi>`,
a function that automatically also determines the character of the scaling.

Finally, there are methods to compute the bounds also in the case of
correlated noise models that have been developed recently in
:cite:`kurdzialek2024bounds`. We do not provide the exact formulation of
the optimization problem that one needs to solve to find the relevant
bound as it is much more involved than in case of uncorrelated noise
models discussed above. In the package, the correlated noise bounds may
be computed using
:py:func:`ad_bounds_correlated <qmetro.bounds.bounds.ad_bounds_correlated>`
a finite number of channel uses, and
:py:func:`ad_asym_bound_correlated <qmetro.bounds.bounds.ad_asym_bound_correlated>`
in order to obtain the asymptotic behavior. Unlike in the uncorrelated
noise models, there is no guarantee here that the bounds are saturable.
Still, they may be systematically tightened at the expense of increasing
the numerical complexity, see Sec. :ref:`sec:correlated` for
a more detailed discussion.
