Polarisation Fractions
----------------------

The following is a list of common ways to calculate the integrated polarisation fraction of a time profile.
We will first define :math:`Q(t), U(t)` to be our Linear polarisation modes and :math:`V(t)` the Circular Polarisation
as a function of time, assuming we have averaged in frequency. :math:`I(t)` is the total power profile. The uncertainty in these 
quantities are :math:`\sigma_{X}`, where :math:`X` is the Stokes parameter. For the following I will assume :math:`Q, U, V` ect. 
are the Stokes time profiles, for readibility. 

Note: when I say "averaged in frequency", I mean integrating over frequency then deviding by the number of channels.
This will matter if you are after the absolute integrated flux density of Linear/Circular polarisation, however, for
calculating fractions, the averaging cancels out.


:math:`L` and :math:`P`
=============================
The linear polarisation is

.. math::
   L = \sqrt{Q^{2} + U^2}.

Given this functional form, the uncertainty in :math:`L` is,

.. math::
   \sigma_{L} = \frac{1}{L}\sqrt{Q^{2}\sigma_{Q}^2 + U^{2}\sigma^{2}}.

there is a need to debias :math:`L` due to the expression used, especially at low signal-to-noise (S/N)

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



:math:`L/I = l` - Linear Polarisation fraction
======================================

.. math::
   \bar{l} = \frac{\sum{(Q^{2} + U^{2})}}{\sum{I}}

Used often in pulsar/FRB literature, such as Sherman et al, 2024. 

.. math::
   l^{*} = \frac{\sqrt{(\sum{Q})^{2} + (\sum{U})^{2}}}{\sum{I}}

This form has mention in Oswald et al, 2023.

The uncertainty in the integrated Stokes profile :math:`\sum{X}` is given by

.. math::
   \sigma_{\sum{X}} = \sqrt{\sum_{i}{\sigma_{X, i}^{2}}}

if the Stokes profile is mean subtracted, i.e. the mean is zero, then the uncertainty in :math:`\sum{X}` is simply the 
standard deviation multiplied by :math:`\sqrt{N}`, where :math:`N` is the number of samples in the Stokes profile, i.e., ``N**0.5 * std(\sigma_{X})``.

The final uncertainty of some fraction :math:`X/I` is

.. math::
   \sigma_{X/I} = \frac{X}{I}\sqrt{\frac{\sigma_{X}^{2}}{X^{2}} + \frac{\sigma_{I}^{1}}{I^{2}}}
.

It's easier to work through these equations one at a time then to construct a big explicit equation for each Stokes fraction. It can
get messy...
   



:math:`V/I = \nu` - Circular Polarisation fraction
==================================================

Cicular polarisation can be a pain since the profile can have different signs (+/-). It can also flip sign (change of handedness)
over time. This there are a few ways of calculating the Circular polarisation fraction,

.. math::
   \bar{\nu} = \frac{\sum{V}}{\sum{I}}

.. math::
   |\nu|^{*} = |\bar{\nu}|

.. math::
   \bar{|\nu|} = \frac{\sum{(|V|\mathrm{_{d}})}}{\sum{I}}.

The last method is used in Oswald et al, 2023. Additionally, :math:`|V|\mathrm{_{d}}` is the debiased absolute Circular polarisation, following
Karastergiou et al. 2023 and Posselt et al. 2022. This method seems to be the most "correct" when measuring the fraction of Circular
polarisation.



:math:`P/I = p` - Total Polarisation fraction (This is the "fun" part)
======================================================================

There are a few ways of calculating :math:`p`.

The simplest and likely most "correct", has the following form

.. math::
   \bar{p} = \frac{\sum{P\mathrm{_{d}}}}{\sum{I}}.

Remember that from the above equations, :math:`L\mathrm{_{d}}}` is used to calculate :math:`P` which is then also debiased.

The second method seperatley calculates the Stokes :math:`Q, U` and :math:`V` polarisation fractions, then adds them in quadrature,

.. math::
   p^{*} = \sqrt{\bigg(\frac{\sum{Q}}{\sum{I}}\bigg)^{2} + \bigg(\frac{\sum{U}}{\sum{I}}\bigg)^{2} + \bigg(\frac{\sum{V}}{\sum{I}}\bigg)^{2}}.

The third method calculates :math:`P/I` by first integrating the absolute Stokes V profile

.. math::
   \bar{p_{|\nu|}} = \sqrt{\bar{l}^{2} + \bar{|\nu|}^{2}}.

This method is less useful, so it's recommended to use the first two.









   
