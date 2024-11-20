Polarisation Fractions
----------------------

The following is a list of polarisation fraction equations used by ILEX to measure to total
frequency-time averaged polarisation fractions within an FRB or any radio profile with full polarisation.

We will first define :math:`Q(t), U(t)` to be our Linear polarisation modes and :math:`V(t)` the Circular Polarisation
as a function of time, assuming we have averaged in frequency. :math:`I(t)` is the total power profile. The uncertainty in these 
quantities are :math:`\sigma_{X}`, where :math:`X` is the Stokes parameter. For the following I will assume :math:`Q, U, V` ect. 
are the Stokes time profiles, for readibility. 

Note: when I say "averaged in frequency", I mean integrating over frequency then deviding by the number of channels.
This will matter if you are after the absolute integrated flux density of Linear/Circular polarisation, however, for
calculating fractions, the averaging cancels out.





:math:`L` and :math:`P`
=============================

We measure the Linear polarisation via the following equation

.. math::
   L = \sqrt{Q^{2} + U^2}.

Given this functional form, the uncertainty :math:`\sigma_{L}` is

.. math::
   \sigma_{L} = \frac{1}{L}\sqrt{Q^{2}\sigma_{Q}^2 + U^{2}\sigma_{U}^{2}}.

there is a need to debias :math:`L` due to the expression used, especially at low signal-to-noise (S/N)
[Everett & Weisberg 2001; Day et al. 2020]

.. math::
   L\mathrm{_{d}} = 
    \begin{cases} 
      \sigma_{I}\sqrt{\bigg(\frac{L}{\sigma_{I}}\bigg)^{2} - 1} & \frac{L}{\sigma_{I}} > 1.57 \\
      0 & \mathrm{otherwise}. \\
   \end{cases}

The error in :math:`L\mathrm{_{d}}` is done in a similar way when calculating the un-debiased Linear polarisation uncertainty :math:`\sigma_{L}`,
but replacing :math:`L` with :math:`L\mathrm{_{d}}`.



Calculating the total polarisation P can be done one of two ways, either by performing a quadratic sum of :math:`Q,U,V`,

.. math::
   P = \sqrt{Q^{2} + U^{2} + V^{2}},

or by using :math:`L\mathrm{_{d}}`

.. math::
   P = \sqrt{L\mathrm{_{d}}^2 + V^{2}}.

The second way is easier, and can easily be debised in the same way as with :math:`L`. The uncertainity is of the same functional
form as for :math:`L` above, but with :math:`Q, U` interchanged with :math:`L, V`, same goes for the debiased total polarisation.





Calculating Polarisation fractions
==================================

We measure the total polarisation fraction by first integrating the Stokes profile and dividing by the total integrated power

.. math::
   x = \frac{\sum{X}}{\sum{I}}

where the lower case :math:`x` denotes the Stokes polarisation fraction. We can also measure the uncertainty :math:`\sigma_{x}`, 
which we can split up into two seperate stages.

1. Calculate the uncertainty in the integrated stokes parameters :math:`\sigma\mathrm{_{\sum{X}}}` and :math:`\sigma\mathrm{_{\sum{I}}}`. 
Which we can do using the following equation

.. math::
   \sigma\mathrm{_{\sum{X}}} = \sqrt(\sum{\sigma_{X}}).

If the error for the entire Stokes profile is a single value, we can estimate the uncertainity using basic noise scaling

.. math::
   \sigma\mathrm{_{\sum{X}}} = \sqrt{N} * \sigma_{X},

where :math:`N` is the number of samples in the Stokes profile.


2. The next step is to calculate the final uncertainity in the stokes polarisation fraction, which is done using the following equation

.. math::
   \sigma_{x} = x\sqrt{\frac{\bigg(\sum{X}}{\sigma\mathrm{_{\sum{X}}}}\bigg)^{2} + \frac{\bigg(\sum{I}}{\sigma\mathrm{_{\sum{I}}}}\bigg)^{2}}

With the above equations, we can calculate all the integrated Stokes polarisation fractions and their uncertainties. Below is the list for
completeness.






Absolute Stokes :math:`V` intergrated polarisation fraction
===========================================================

One important aspect of Circular polarisation we have to keep in mind is that it is a signed quantity, meaning is can be +/-. It may also
flip between being + or - across the profile. So simply integrating over the profile is not a reliable measure of the Stokes :math:`V`
polarisation fraction.

We can calculate the Stokes :math:`V` polarisation fraction by first taking the absolute value of the Stokes profile before integrating

.. math::
   |v| = \frac{\sum{|V|}}{\sum{I}}.

Before integrating the absolute stokes :math:`V` profile, :math:`|V|` needs to be debiased according to Karastergiou et al. 2003, Posselt et al. 2022 
and Oswald et al. 2023

.. math::
   |V|\mathrm{_{d}} = 
   \begin{cases} 
      |V| - \sigma_{I}\sqrt{\frac{2}{\pi}} & |V| > \sigma_{I}\sqrt{\frac{2}{\pi}} \\
      0 & \mathrm{otherwise}.
   \end{cases}

thus the absolute integrated Stokes :math:`V` polarisation fraction is

.. math::
   |v| = \frac{\sum{|V|\mathrm{_{d}}}}{\sum{I}}.


Note of :math:`|q|` and :math:`|u|`
+++++++++++++++++++++++++++++++++++

The same arguments can be made about stokes :math:`Q` and :math:`U` since they can also change sign across time. The above equations may be
used interchangably, although :math:`|q|` and :math:`|u|` are less useful.








Continuum-added polarisation fractions
======================================

The complete set of integrated polarisation fractions are listed below for completeness. They will also be useful when using other methods for
calculating the Linear and total integrated polarisation fractions.

.. math::
   \begin{split}
      & |q| = \frac{\sum{|Q|\mathrm{_{d}}}}{\sum{I}} \\
      & |u| = \frac{\sum{|U|\mathrm{_{d}}}}{\sum{I}} \\
      & |v| = \frac{\sum{|V|\mathrm{_{d}}}}{\sum{I}} \\
      & l = \frac{\sum{L\mathrm{_{d}}}}{\sum{I}} \\
      & p = \frac{\sum{P\mathrm{_{d}}}}{\sum{I}}
   \end{split}

Signed integrated stokes polarisation fractions

.. math::
   \begin{split}
      & q = \frac{\sum{Q}}{\sum{I}} \\
      & u = \frac{\sum{U}}{\sum{I}} \\
      & v = \frac{\sum{V}}{\sum{I}}.
   \end{split}











Vector-added polarisation fractions
===================================

Alternative methods for calculating :math:`l` and :math:`p` by first intregrating Stokes :math:`Q, U` and :math:`V`, then adding in
quadrature. 

.. math::
   \begin{split}
      & l* = \sqrt{q^{2} + u^{2}} \\
      & p* = \sqrt{q^{2} + u^{2} + v^{2}} \\
      & |l|* = \sqrt{|q|^{2} + |u|^{2}} \\
      & |p|* = \sqrt{|q|^{2} + |u|^{2} + |v|^{2}}.
   \end{split}











Vector-added continuum integrated Stokes :math:`P` polarisation fraction
========================================================================

Alternative method for calculating :math:`p` by adding :math:`l` and :math:`v` in quadrature.

.. math::
   \begin{split}
      & \hat{p} = \sqrt{l^{2} + v^{2}} \\
      & \hat{|p|} = \sqrt{l^{2} + |v|^{2}}.
   \end{split}











