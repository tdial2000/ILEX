Advanced functions
------------------

Time series and time scatter Fitting
====================================

Following the last tutorial, load in the Power dynamic spectra data of 220610 and define a crop region.

.. code-block:: python

    from ilex.frb import FRB

    # initialise FRB instance and load data
    frb = FRB(name = "FRB220610", cfreq = 1271.5, bw = 336, dt = 50e-3, df = 4, t_crop = [20.9, 23.8],
                f_crop = [1103.5, 1200])
    frb.load_data(ds_I = "examples/220610_dsI.npy")  

We can fit the time series burst as a sum of Gaussian pulses convolved with a common one-sided exponential

.. math::
   I(t) = \sum_{i = 1}^{N}\bigg[A_{i}e^{-(t-\mu_{i})^{2}/2\sigma_{i}^{2}}\bigg] * e^{-t/\tau},

where :math:`A_{i}, \mu_{i}` and :math:`\sigma_{i}` are the amplitude, position in time and pulse width 
in time of each :math:`i^{\mathrm{th}}` Gaussian and :math:`'*'` denotes the convolution operation.

This is implemented in the ``.fit_tscatt()`` method of the ``FRB`` class. The simplest way to call this 
function is to use a ``least squares`` fitting method. 

.. code-block:: python

    # fit
    frb.fit_tscatt(method = "least squares", show_plots = True)

.. image:: 220610_tI_fit1.png
   :width: 720pt

In most cases, an FRB burst will be more complicated. In which case a more robust method using the ``bayesian``
toggle is nessesary. To do so, priors need to be given. We also need to give our best estimate for the number
of pulses in the burst, which we can do with ``npulse``

.. code-block:: python

    # priors
    priors = {'a1': [0.5, 0.8], 'mu1': [21.0, 22.0], 'sig1': [0.1, 1.0], 'tau': [0.01, 2.0]}

    # fit
    p = frb.fit_tscatt(method = "bayesian", priors = priors, npulse = 1, show_plots = True)

.. image:: 220610_tI_fit2.png
   :width: 720pt

In the above code, we set the priors of the single pulse with suffixes ``1``, i.e. ``a1`` for the amplitude of the 
first pulse, ``mu1`` for the position of the first pulse etc. If we had two pulses, we would also give priors for the amplitude
``a2``, position ``mu2`` etc. In general for each pulse ``N``, we specify its parameters ``aN, muN, sigN``. 
We can also return the ``p`` object, which is a fitting utility class which has a number of useful features. Most notable is showing
the stats of the modelling.

.. code-block:: python

    p.stats()

.. code-block:: console

    Model Statistics:
    ---------------------------
    chi2:                         52.2002   +/- 10.2956
    rchi2:                        0.9849    +/- 0.1943
    p-value:                      0.5053
    v (degrees of freedom):       53
    free parameters:            5

    Bayesian Statistics:
    ---------------------------
    Max Log Likelihood:           127.7476  +/- 2.2624
    Bayes Info Criterion (BIC):   -235.1929 +/- 4.5248
    Bayes Factor (log10):         nan
    Evidence (log10):             48.0135   +/- 0.0980
    Noise Evidence (log10):       nan

Fitting RM and plotting Position Angle (PA) Profile
===================================================

We can fit for the rotation measure (RM). There are two common methods for doing this.
1. Q/U fitting using the quadratic form of the polarisation position angle (PA)

.. math::
   \mathrm{PA(\nu) = RMc^{2}}\bigg(\frac{1}{\nu^{2}} - \frac{1}{\nu_{0}^{2}}\bigg),

where :math:`\nu_{0}` is the reference frequency. If this is not set, the central frequency ``cfreq``
will be used instead. 

2. Faraday Depth fitting through RM synthesis using the ``RMtools`` package 

https://github.com/CIRADA-Tools/RM-Tools


First we load in the stokes ``Q`` and ``U`` dynamic spectrum.

.. code-block:: python

    # load in data
    frb.load_data(ds_Q = "examples/220610_dsQ.npy", ds_U = "examples/220610_dsU.npy")


We can then fit for the RM using ``.fit_RM()``. We can specify the method to do so
``method = "RMquad"`` for Q/U fitting with a quadratic function.
``method = "RMsynth"`` for RM synthesis.

.. code-block:: python

    # fit RM
    frb.fit_RM(method = "RMsynth", terr_crop = [0, 15], t_crop = [21.4, 21.6], show_plots = True)

.. code-block:: console

    Fitting RM using RM synthesis
    RM: 217.9462  +/-  4.2765     (rad/m2)
    f0: 1137.0805274869874    (MHz)
    pa0:  1.0076283903583936     (rad)

.. image:: 220610_RM.png
   :width: 720pt

The ``RM``, ``f0`` and ``pa0`` parameters will be saved to the ``.fitted_params`` attribute of the ``FRB`` class.
Once RM is calculated, we can plot a bunch of polarisation properties using the master method ``.plot_PA()``.

.. code-block:: python

    frb.set(RM = 217.9462, f0 = 1137.0805274869874)
    frb.plot_PA(terr_crop = [0, 15], stk2plot = "ILV", show_plots = True)

.. image:: 220610_PA.png
   :width: 720pt


Weighting data
==============

One Useful feature of ILEX is weighting. The ``frb.par.tW`` and ``frb.par.fW`` attributes are ``weights`` class instances that
can be used to respectivley weight data in time when making spectra, or weight data in frequency when making time profiles. The 
``weights`` class found in ``ilex.par`` has many methods for making weights, we will use ``method = func`` which will allow us
to define a weighting function. The plots below show the before and after of applying a set of time weights before scrunching in
time to form a spectra of stokes I.

.. code-block:: python

    # lets make a simple scalar weight that multiplies the samples in time
    # by -1 so we can see it works
    # lets plot the before and after 
    frb.plot_data("fI")     # before

    frb.par.tW.set(W = -1, method = "None")
    frb.plot_data("fI")     # after
    # NOTE: the None method is used to specify we want to take the values weights.W as 
    # the weights

.. image:: spec_before_W.png
   :width: 720pt

.. image:: spec_after_W.png
   :width: 720pt


We can be a little more creative with how we define our weights. Lets define a function based on the posterior of our time 
series profile we fitted before.

.. code-block:: python

    # import function to make scattering pulse function
    from ilex.fitting import make_scatt_pulse_profile_func

    # make scatt function based on number of pulses, in this case 1
    profile = make_scatt_pulse_profile_func(1)

    # define a dictionary of the posteriors of the fiting
    args = {'a1': 0.706, 'mu1': 21.546, 'sig1': 0.173, 'tau': 0.540}

    # another method of setting the weights in either time or frequency (xtype)
    frb.par.set_weights(xtype = "t", method = "func", args = args, func = profile)

    # now weight, The rest is left to you, why not plot it?

