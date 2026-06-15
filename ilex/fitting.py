##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 25/09/2023 
##
##
## 
## 
## Stats functions for analysing FRBs.
##
##
##===============================================##
##===============================================##
# imports
#TODO: add import guards?
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import correlate
import bilby
from copy import deepcopy
import sys, inspect
from .logging import log
from math import ceil, floor
import matplotlib.pyplot as plt


# rm synthesis
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean

## import utils ##
from .utils import struct_

from .data import *

from .pyfit import fit, _clean_bilby_run, _priorUniform, PyfitLikelihood

from .globals import *

class _empty:
    pass


##===============================================##
##             utility functions                 ##
##===============================================##


def model_curve(y, n: int = 5, samp: int = None):
    """
    Fit Polynomial model to data.

    Parameters
    ----------
    y : np.ndarray
        data to model
    n : int, optional
        Polynomial order, by default 5
    samp : int, optional
        number of samples to sample modelled data, if None, then any NaN values will be preserved, else NaN values are removed, by default None

    Returns
    -------
    np.ndarray
        Modelled data
    """    
    yout = y.copy()
    ytemp = y[~np.isnan(y)]

    x = np.linspace(0, 1.0, ytemp.size)    
    if samp is None:
        samp = x.size
    xnew = np.linspace(0, 1.0, samp)
    
    y_fit = np.poly1d(np.polyfit(x,ytemp,n))

    if samp != x.size:
        yout[~np.isnan(y)] = y_fit(xnew)
    else:
        yout = y_fit(xnew)

    return yout





def make_polyfit(n):

    mstr = "lambda x"
    mstr2 = ""

    for i in range(n):
        mstr += f", x{i}"
        mstr2 += f"x{i} * x**{n-i} + "
    
    mstr += ", c : "
    mstr2 += "+ c"

    return eval(mstr + mstr2)




##===============================================##
##           basic fitting functions             ##
##===============================================##

## [ LORENTZIAN FUNCTION ] ##
def lorentz(x,w,a):
    """
    Lorentz function - Usually used to model scintillation bandwidth

    Parameters
    ----------
    x : np.ndarray
        X data - Usually Frequency array
    w : float
        width - Usually Scintillation Bandwidth
    a : float
        Amplitude - Usually m^2 where m is modulation index

    Returns
    -------
    np.ndarray
        Y data
    """

    return a*w**2/(w**2+x**2)

def lorentz_yshifted(x, w, a, h):
    """
    Lorentz function - Usually used to model scintillation bandwidth

    Parameters
    ----------
    x : np.ndarray
        X data - Usually Frequency array
    w : float
        width - Usually Scintillation Bandwidth
    a : float
        Amplitude - Usually m^2 where m is modulation index

    Returns
    -------
    np.ndarray
        Y data
    """

    return a*w**2/(w**2+x**2) + h



## [ GAUSSIAN FUNCTION ] ##
def gaussian(x, a, mu, sig):
    """
    Gaussian Pulse Function

    Parameters
    ----------
    x : np.ndarray
        X data
    a : float
        amplitude
    mu : float
        position of Gaussian Pulse
    sig : float
        width of Gaussian Pulse

    Returns
    -------
    np.ndarray
        Y data 
    """

    return a*np.exp(-(x-mu)**2/(2*sig**2))



def scat(dx, tau, sig = 10):
    """
    1 sided (positive side) exponential Scattering tail function

    Parameters
    ----------
    x : np.ndarray
        X data
    tau : float
        Scattering Timescale
    sig: float
        number of standard deviations for defined scat function from mean

    Returns
    -------
    np.ndarray
        Y data
    """

    # create x with same time resolution
    _w = int(ceil(tau*sig/dx)) 
    x = np.linspace(-_w*dx, _w*dx, 2*_w+1)
    hw = x.size//2
    yscat = np.zeros(x.size)

    # only fill one side, since this is a one sided
    # exponential
    yscat[hw:] = np.exp(-x[hw:]/(tau))
    return yscat







def specindex(x, a, alpha):
    """
    Spectral index power-law function

    Parameters
    ----------
    x : np.ndarray
        X data
    a : float
        Amplitude
    alpha : float
        Power-law index

    Returns
    -------
    np.ndarray
        Y data
    """

    # spectral index function
    return a*x**alpha



def specindex2(x, tau_c, alpha):
    """
    Method 2 of calculating tau and the spectral index power-law
    function. Better for fitting.

    tau(f) = tau_c * (f/f_c)**alpha

    Parameters
    ----------
    x : np.ndarray
        Frequency [MHz]
    tau_c : float
        Scattering timescale at the central frequency f_c
    alpha : float
        Power-law index
    
    f_c is estimated using [x] -> f_c = (x[0] + x[-1])/2
    """

    f_c = (x[0] + x[-1])/2

    return tau_c * (x / f_c)**alpha




def burnslaw(x, p, sig_rm):
    """
    Calculate modified burns law

    Parameters
    ----------
    x : np.ndarray
        Frequency [MHz]
    """
    # return np.exp(-2*sig_rm**2*(c/x*1e-6)**4)
    return p * np.exp(-2*sig_rm**2*(c/x*1e-6)**4)



##===============================================##
##          Advanced fitting functions           ##
##===============================================##
def scatt_pulse_profile(x, **p):
    """
    Scattering time series profile with n pulses. Numerical convolution if done incorrectly
    can shift the resultant data in an undesirable way. One way to avoid this is to take a large window
    around the known signal to encompass the all pulses and convolve this with a symmetrical scattering tail.
    This of course isn't realistic when taking a crop of data whose bounds cut through potential signal. 

    To keep this function robust, the algorithm implemented here takes each gaussian profile and extends it until 
    symmetrical, this avoids any potential shifting due to improper convolution. 

    Parameters
    ----------
    x: np.ndarray
        X data array
    p: Dict(float)
        dictionary of parameters for scattered Gaussian pulses, for each pulse n: \n
        [a[n]] - Pulse amplitude \n
        [mu[n]] - Pulse position \n
        [sig[n]] - Pulse width \n
        [tau] - scattering timescale

    Returns
    -------
    y: np.ndarray
        Y data array
    """
    # create empty output array
    y = np.zeros(x.size)
    
    # create scattering tail 
    dt = x[1] - x[0]
    npulses = (len(p) - 1)//3
    stail = scat(dt,p['tau'])

    # Each gaussian will be isolated and convolved seperatley with enough padding for a complete
    # uniform convolution with zero shifting due to numerical error.
    for i in range(npulses):

        # make gaussian with sigma 5
        xe = int(floor((p[f"mu{i+1}"] + p[f"sig{i+1}"]*5)/dt))
        xs = int(floor((p[f"mu{i+1}"] - p[f"sig{i+1}"]*5)/dt))
    
        # make sure the scattering tail is smaller or equal to the size of the gaussian to convolve
        if xe-xs + 1 < stail.size:
            # expand to same size as stail
            lendif = int(ceil((stail.size - (xe-xs+1))/2))
            xe += lendif
            xs -= lendif
            
            
        x_i = np.linspace(xs*dt, xe*dt, xe-xs + 1)
            
        # crop bounded signal
        ps = int(floor(x[0]/dt))
        xs -= ps
        xe -= ps

        # handle edge cases
        if xs >= x.size:
            continue

        if xe <= 0:
            continue

        # make pulse
        pulse_i = gaussian(x_i, 1, p[f"mu{i+1}"], p[f"sig{i+1}"])

        # convolve
        conv = np.convolve(pulse_i, stail, mode = "same")
        pulse_ind = [0, conv.size]
        
        if xs < 0:
            pulse_ind[0] = 0 - xs
            xs = 0
        if xe + 1 > x.size:
            pulse_ind[1] -= (xe+1 - x.size)
            xe = x.size

        y[xs:xe+1] += p[f"a{i+1}"] * conv[pulse_ind[0]:pulse_ind[1]]/np.max(conv)

    return y



def scatt_pulse_profile_relative(x, **p):
    """
    The functionality and inputs are the exact same to ilex.fitting.scatt_pulse_profile,
    the difference is that [mu2, mu3, mu4, ..., muN] are treated as relative positions from 
    mu1.
    """

    # convert relative positions to absolute positions
    pnew = deepcopy(p)
    npulses = (len(pnew) - 1)//3
    if npulses > 1:
        for i in range(2, npulses+1):
            pnew[f'mu{i}'] += pnew['mu1']
    return scatt_pulse_profile(x, **pnew)





# def _scatt_pulse_profile_func():
#     pass

def make_scatt_pulse_profile_func(n = 1):
    """
    Make scatter pulse profile wrapping function for fitting

    Parameters
    ----------
    n: int
        number of pulses in scatter profile
    Returns
    -------
    func: __func__
        lambda function for scatter pulse profile with n pulses, only if return_definition == False

    """

    args_str = "lambda x"
    func_str = "scatt_pulse_profile(x,"

    for i in range(1, n+1):
        # loop through components
        for p in ["a", "mu", "sig"]:
            func_str += f"{p}{i} = {p}{i},"
            args_str += f",{p}{i}"

    # add tau
    func_str += "tau = tau"
    args_str += ",tau"

    return (eval(args_str + ":" + func_str + ")"))





def make_scatt_pulse_profile_relative_func(n = 1):
    """
    Make scatter pulse profile wrapping function for fitting

    Parameters
    ----------
    n: int
        number of pulses in scatter profile
    Returns
    -------
    func: __func__
        lambda function for scatter pulse profile with n pulses, only if return_definition == False

    """

    args_str = "lambda x"
    func_str = "scatt_pulse_profile_relative(x,"

    for i in range(1, n+1):
        # loop through components
        for p in ["a", "mu", "sig"]:
            func_str += f"{p}{i} = {p}{i},"
            args_str += f",{p}{i}"

    # add tau
    func_str += "tau = tau"
    args_str += ",tau"

    return (eval(args_str + ":" + func_str + ")"))







def make_scatt_pulse_profile_func_multi(n = 1):
    """
    Make scatter pulse profile wrapping function with the option of specifying more than 1
    scattering timescales
    
    Parameters
    ----------
    n: int
        number of scattering timescales
    Returns
    -------
    func: __func__
        lambda function of the form func(x, args1, tau1, args2, tau2...), where args are the set
        of gaussian parameters args1 = {'mu1', 'sig1' ...}. Each set of args can have any number
        of gaussian paramters and will be convolved with it's corrosponding tau1. The final result is the addition
        of these scatt pulse profiles, i.e. I = scatt_pulse_profile(x, tau = tau1, **args1) + scatt_pulse_profile(x, tau = tau2, **args2) + ...

    """

    args_str = "lambda x"
    func_str = ""

    for i in range(1, n+1):
        args_str += f", args{i}, tau{i}"
        func_str += f"scatt_pulse_profile(x, tau = tau{i}, **args{i}) + "
    
    return(eval(args_str + ":" + func_str[:-3]))






def make_scatt_pulse_profile_func_relative_multi(n = 1):
    """
    Make scatter pulse profile wrapping function with the option of specifying more than 1
    scattering timescales
    
    Parameters
    ----------
    n: int
        number of scattering timescales
    Returns
    -------
    func: __func__
        lambda function of the form func(x, args1, tau1, args2, tau2...), where args are the set
        of gaussian parameters args1 = {'mu1', 'sig1' ...}. Each set of args can have any number
        of gaussian paramters and will be convolved with it's corrosponding tau1. The final result is the addition
        of these scatt pulse profiles, i.e. I = scatt_pulse_profile(x, tau = tau1, **args1) + scatt_pulse_profile(x, tau = tau2, **args2) + ...

    """

    args_str = "lambda x"
    func_str = ""

    for i in range(1, n+1):
        args_str += f", args{i}, tau{i}"
        func_str += f"scatt_pulse_profile_relative(x, tau = tau{i}, **args{i}) + "
    
    return(eval(args_str + ":" + func_str[:-3]))







class tscattLikelihood(PyfitLikelihood):
    def __init__(self, x, y, yerr, npulse, fitmode = "abs"):
        """
        Likelihood function for fitting N pulses with a common scattering
        exponential to FRB data.

        Parameters
        ----------
        x : np.ndarray
            Time [ms]
        y : np.ndarray
            FRB data
        yerr : np.ndarray
            error in y
        npulse : np.ndarray
            Number of gaussian pulses to model FRB
        """
        # set data attributes
        self.npulse = npulse

        # construct parameters, we aren't going to retrieve them from the 
        # function because we need the function to accept a dictionary of parameters
        # of variable length given self.npulse. So we will construct the parameters here
        parameters = ['tau']
        for n in range(1, npulse+1):
            parameters += [f'a{n}', f'mu{n}', f'sig{n}']
        bilby.Likelihood.__init__(self, parameters = dict.fromkeys(parameters))
        self.parameters = dict.fromkeys(parameters)
        self.func_keys = list(self.parameters.keys())

        if fitmode == "abs":
            func = scatt_pulse_profile
        elif fitmode == "relative":
            func = scatt_pulse_profile_relative
        else:
            ValueError(f"fitmode = {fitmode} is not a valid method of fitting tscatt!")

        self._superinit(x = x, y = y, func = func, yerr = yerr)


    def log_likelihood(self):
        """
        Evaluate the log likelihood function using a dictionary to pass to scatt_pulse_profile()
        
        """

        sigma = self.sigma
        y_model = self.func(self.x, **self.model_parameters)

        # for a gaussian log likelihood
        return -0.5 * np.sum(((self.y - y_model)/sigma)**2 + np.log(2 * np.pi * sigma**2))
















##============================##
##          fitting           ##
##============================##



## RM fitting functions ##
def fit_RMsynth(I, Q, U, Ierr, Qerr, Uerr, f, Inorm = True, clean_cutoff = 0.1, **kwargs):

    """
    Use RM synthesis to calculate RM, pa0 and f0,
    f0 is the weighted midband frequency and pa0 the
    pa at f0.

    Parameters
    ----------
    I: np.ndarray
        stokes I spectra
    Q: np.ndarray 
        stokes Q spectra
    U: np.ndarray 
        stokes U spectra
    Ierr: np.ndarray 
        stokes I rms spectra
    Qerr: np.ndarray 
        stokes Q rms spectra
    Uerr: np.ndarray 
        stokes U rms spectra
    f: np.ndarray 
        frequencies [MHz]
    lnorm: bool 
        if True will normalize stokes parameters by Stokes I, by default True
    clean_cutoff: float 
        cutoff arg for run_rmclean()
    **kwargs: Dict 
        keyword arguments for RM tools run_synthesis

    Returns
    -------
    out : dict
        dictionary of results \n
        rm [float]: rotation measure \n 
        rm_err [float] error in rotation measure \n
        f0 [float] reference frequency at weighted mid-band \n
        pa0 [float] position angle at f0 \n
        phiArr [np.ndarray] Array of phi values values \n
        fdfArr [np.ndarray] Array of faraday depth values
    """

    print(Inorm)

    defkwargs = {"polyOrd":3, "phiMax_radm2":1.0e3, "dPhi_radm2":1.0, "nSamples":100.0, 
                 'showPlots':False}

    ## process kwargs keys
    keys = kwargs.keys()
    for key in defkwargs.keys():
        if key not in keys:
            kwargs[key] = defkwargs[key]


    # get stokes fractions
    if Inorm:
        print("Normalizing Stokes parameters by Stokes I, i.e. Q/I, U/I")
        Qfr, QfrErr = calc_ratio(I, Q, Ierr, Qerr)
        Ufr, UfrErr = calc_ratio(I, U, Ierr, Uerr)


    log("Fitting RM using RM synthesis", lpf = False)
    if not Inorm:
        #RM data array
        rmsyn_data = np.array(
            [
            f * 1e6,     # freqs (Hz)  
            I,           # I
            Q,           # Q
            U,           # U
            Ierr,        # I rms
            Qerr,        # Q rms
            Uerr         # U rms
            ]
        )
    else:
        rmsyn_data = np.array(
            [
            f * 1e6,     # freqs (Hz)  
            Qfr,           # Q fraction
            Ufr,           # U fraction
            QfrErr,        # Q rms fraction
            UfrErr         # U rms fraction
            ]
        )

    # run RM synthesis
    rm_sum, rm_data = run_rmsynth(rmsyn_data, **kwargs)

    # apply RM cleaning
    rmc, rm_data = run_rmclean(rm_sum, rm_data, clean_cutoff, showPlots = kwargs['showPlots'])

    # get estimated parameters
    rm = rmc['phiPeakPIfit_rm2']                                             # RM
    rm_err = rmc['dPhiPeakPIfit_rm2']                                        # RM err
    f0 = rm_sum['freq0_Hz'] / 1e6                                               # f0 (MHz)
    pa0 = 0.5 * np.arctan2(rmc['peakFDFimagFit'],rmc['peakFDFrealFit'])   # pa0 (at f0)

    # print
    log(f"RM: {rm:.4f}  +/-  {rm_err:.4f}     (rad/m2)", lpf = False)
    log(f"f0: {f0}    (MHz)", lpf = False)
    log(f"pa0:  {pa0}     (rad)", lpf = False)

    out = {}
    out['rm'] = rm 
    out['rm_err'] = rm_err
    out['f0'] = f0
    out['pa0'] = pa0
    out['phiArr'] = rm_data['phiArr_radm2'].copy()
    out['fdfArr'] = np.abs(rm_data['cleanFDF'])

    return out














def fit_RMquad(Q, U, Qerr, Uerr, f, f0, **kwargs):
    """
    Info:
        Use Quadratic method to fit for RM and pa0.

    Parameters
    ----------
    Q: np.ndarray 
        stokes Q spectra
    U: np.ndarray 
        stokes U spectra
    Qerr: np.ndarray 
        stokes Q rms spectra
    Uerr: np.ndarray 
        stokes U rms spectra
    f: np.ndarray 
        frequencies [MHz]
    f0: float
        reference Frequency [MHz]
    **kwargs: Dict 
        keyword arguments for RM tools run_synthesis

    Returns
    -------
    out : dict
        Dictionary of results
        rm [float] rotation measure \n
        rm_err [float] error in rotation measure \n
        pa0 [float] position angle at f0 \n
        pa0_err [float] position angle err
    """

    log("Fitting using RM quadratic function", lpf = False)
    # fit RM using Quadratic function
    def rmquad(f, rm, pa0):
        return pa0 + rm*c**2/1e12*(1/f**2 - 1/f0**2)
    

    PA_meas, PA_err = calc_PA(Q, U, Qerr, Uerr)
    PA_meas = np.unwrap(PA_meas, period = np.pi)

    # fit
    fit_val, fit_err = curve_fit(rmquad, f, PA_meas, sigma = PA_err, absolute_sigma = True, **kwargs, maxfev = 2000000)
    fit_err = np.sqrt(np.diag(fit_err))
    
    # get params
    rm = fit_val[0]
    rm_err = fit_err[0]
    pa0 = fit_val[1]
    pa0_err = fit_err[1]

    # print
    log(f"RM: {rm:.4f}  +/-  {rm_err:.4f}     (rad/m2)", lpf = False)
    log(f"f0: {f0}    (MHz)", lpf = False)
    log(f"pa0:  {pa0}  +/-  {pa0_err:.4f}     (rad)", lpf = False)

    out = {}
    out['rm'] = rm
    out['rm_err'] = rm_err
    out['pa0'] = pa0
    out['pa0_err'] = pa0_err

    return out






# QUfitting likelihood class
class QUfit_likelihood(bilby.Likelihood):
    def __init__(self, f, Q, U, Ierr, Qerr, Uerr):
        """
        Likelihood function for evaluating Stokes Q and U parameters to fit RM. L will be debiased
        and a mask will be applied to all Stokes parameters based on L debiased.

        Parameters
        ----------
        f : np.ndarray
            Frequencies in MHz
        Q : np.ndarray
            Stokes Q parameter
        U : np.ndarray
            Stokes U parameter
        Ierr : np.ndarray
            Stokes I parameter noise, used to calculate L debiased
        Qerr : np.ndarray
            Stokes Q parameter noise
        Uerr : np.ndarray
            Stokes U parameter noise
        """

        # constants
        self.c = 299_792_458    # [m/s^2]

        # calculate debiased L
        L_meas = np.sqrt(Q**2 + U**2)
        self.L = Ierr * np.sqrt((L_meas/Ierr)**2 - 1)
        self.L[L_meas/Ierr < 1.57] = np.nan 
        self.mask = ~np.isnan(self.L)
        self.L = self.L[self.mask] 
 
        # f data
        self.f = f[self.mask] * 1e6        # in Hz
        self.N = f.size
        
        # stk data 
        self.Q = Q[self.mask]
        self.U = U[self.mask]

        # stk noise
        self.Qerr = Qerr[self.mask]
        self.Uerr = Uerr[self.mask]

        # These lines of code infer parameters from provided function
        parameters = inspect.getfullargspec(self.PA).args[1:]
        super().__init__(parameters = dict.fromkeys(parameters))
        self.parameters = dict.fromkeys(parameters)

        self.function_keys = ["RM", "pa0"]


    @property
    def model_parameters(self):

        return {k: self.parameters[k] for k in self.function_keys}

    
    def PA(self, RM, pa0):
        """
        PA function to evaluate

        Parameters
        ----------
        RM : float
            Rotation Measure [rad/m^2]
        pa0 : float
            Initial reference polarisation position angle [rad]
        """

        return pa0 + RM * self.c**2 / self.f**2

    

    def log_likelihood(self):
        """
        Log likelihood, adding Q and U likelihoods together
        
        """

        PA = self.PA(**self.model_parameters)

        # calculate Stokes Q log likelihood
        ll_Q = -0.5 * np.sum(np.log(2 * np.pi * self.Qerr**2) + ((self.Q - self.L * np.cos(2*PA))/self.Qerr)**2)

        # calculate Stokes U log likelihood
        ll_U = -0.5 * np.sum(np.log(2 * np.pi * self.Uerr**2) + ((self.U - self.L * np.sin(2*PA))/self.Uerr)**2)

        return ll_Q + ll_U




def RM_QUfit(Q, U, Ierr, Qerr, Uerr, f, rm_priors = [-1000, 1000], pa0_priors = [-3.1415926, 0], **kwargs):
    """
    Fit RM using QUfit method

    Parameters
    ----------
    f : np.ndarray
        Frequencies in MHz
    Q : np.ndarray
        Stokes Q parameter
    U : np.ndarray
        Stokes U parameter
    Ierr : np.ndarray
        Stokes I parameter noise, used to calculate L debiased
    Qerr : np.ndarray
        Stokes Q parameter noise
    Uerr : np.ndarray
        Stokes U parameter noise

    Returns
    -------
    out : dict
        Dictionary of results \n
        rm [float] rotation measure \n
        rm_err [float] error in rotation measure \n
        pa0 [float] position angle at f0 \n
        pa0_err [float] position angle err
    """

    outdir = "outdir"
    if "outdir" in kwargs.keys():
        outdir = kwargs["outdir"]
    label = "label"
    if "label" in kwargs.keys():
        label = kwargs["label"]
    _clean_bilby_run(outdir, label)


    priors = {'RM':rm_priors.copy(), 'pa0':pa0_priors.copy()}

    # start sampling
    likelihood = QUfit_likelihood(f = f, Q = Q, U = U, Ierr = Ierr, Qerr = Qerr, Uerr = Uerr)

    result = bilby.run_sampler(likelihood = likelihood, priors = _priorUniform(priors),
                                **kwargs)

    # get rm and pa0 measurements
    rm_posterior = result.get_one_dimensional_median_and_error_bar('RM')
    rm = rm_posterior.median
    rm_err = (abs(rm_posterior.plus) + abs(rm_posterior.minus))/2

    pa0_posterior = result.get_one_dimensional_median_and_error_bar('pa0')
    pa0 = pa0_posterior.median
    pa0_err = (abs(pa0_posterior.plus) + abs(pa0_posterior.minus))/2

    # plots 
    result.plot_corner()

    out = {}
    out['rm'] = rm
    out['rm_err'] = rm_err
    out['pa0'] = pa0
    out['pa0_err'] = pa0_err

    return out

    
