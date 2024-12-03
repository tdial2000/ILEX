Scattering and Scintillation
----------------------------




Scattering
==========

We do not nessesarily know (as of writing this, 2/12/2024) the instrinsic emission of FRBs. Hence when we go to fit their
profiles we need to make some assumption. The most widely used and resonable basis function is a Gaussian pulse

.. math::
   f(t) = Ae^{-(t - \mu)^{2}/2\sigma^{2}}.

Where :math:`A, \mu` amd :math:`\sigma` are the amplitude, position and width (standard deviation) of the pulse.
An FRB can have multiple components, so usually we model the instrinic emission of an FRB as a sum of gaussian components.
However, the observed signal will have undergone some level of scattering as it propogated through ionised plasma along the line of 
sight. This can be modelled simply using a one-sided exponential (where the 'one-sided' means ill-defined for :math:`t < 0`). This scattering
function is then convolved with the intrinsic FRB emission to get the observed FRB emission

.. math::
    I(t) = \sum_{i = 1}^{N}\bigg[A_{i}e^{-(t-\mu_{i})^{2}/2\sigma_{i}^{2}}\bigg] * e^{-t/\tau_{s}}.

Here the FRB emission is made up of :math:`N` Gaussian pulses, which have been scattered by a timescale of :math:`\tau_{s}` (scattering timescale).





Scintillation
=============

Scintillation is the phenomena where incoming radio emission undergoes a phase of focusing and defocusing due to large/small scale inhomogenities
in the intervening ionised plasma. This is a frequency dependant effect and results in "patchy" structures in the radio emission spectra.

Modelling scintillation in FRBs is most commonly done by modelling the 1D ACF (Auto-correlation function) of the spectrum as a lorentzian function

.. math::
   \mathrm{ACF}(\Delta \nu) = m^{2}\frac{\nu_{dc}^{2}}{\nu_{dc}^{2} + \Delta \nu^{2}},

where :math:`\Delta \nu` is the frequency lag in the ACF and :math:`\nu_{dc}` is the decorrelation bandwidth.


   