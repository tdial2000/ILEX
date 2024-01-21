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

# rm synthesis
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean

## import utils ##
from .utils import struct_

from .data import *

# constants
c = 2.997924538e8 # Speed of light [m/s]


class fit_par:

    def __init__(self):

        self.val = {}
        self.plus = {}
        self.minus = {}
        self.keys = []

        pass


    def result2par(self, result):
        """
        Info:
            Get posterior info from BILBY result class

        Args:
            result (result): BILBY result class

        Returns:
            p (dict): Dictionary of posterior params
        """
        # get all parameters excluding log_likelihood
        par_keys = result.posterior.columns[:-2].tolist()

        for key in par_keys:
            vals = result.get_one_dimensional_median_and_error_bar(key)
            self.val[key] = vals.median
            self.plus[key] = vals.plus
            self.minus[key] = vals.minus
            self.keys.append(key)

    

    def static2par(self, dic):
        """
        Info:
            Get static params infomation from dictionary
            and add to par class

        Args:
            dic (dict): Static prior dict
        """

        for key in dic.keys():
            self.val[key] = dic[key]
            self.plus[key] = 0.0
            self.minus[key] = 0.0
            self.keys.append(key)


        
    def add_param(self, key, val, plus, minus):
        """
        Info:
            Add param to container, if param exists, will be overwritten 
            instead.

        Args:
            key (str): parameter name
            val (float): bestfit value
            plus (float): positive error
            minus (float): negative error

        """

        if key not in self.keys:
            self.keys.append(keys)

        self.val[key] = val
        self.plus[key] = plus
        self.minus[key] = minus



    

    def get_bestfit(self):
        """
        Info:
            Get parameter bestfit values

        Args:
            vals (dict): bestfit values

        """
        bestfit = {}
        params = deepcopy(self.val)

        for key in self.keys:
            if key != "sigma":
                bestfit[key] = params[key]

        return bestfit




    def get_err(self):
        """
        Info:
            Get parameter errors values

        Args:
            plus (dict): positive errors
            minus (dict): negative errors
            
        """

        plus = {}
        minus = {}
        plus_par = deepcopy(self.val)
        minus_par = deepcopy(self.val)

        for key in self.keys:
            if key != self.keys:
                plus[key] = plus_par[key]
                minus[key] = minus_par[key]

        return plus, minus




    def get_keys(self):
        """
        Info:
            Get parameter keys

        Args:
            keys (dict): parameter keys

        """

        keys = []

        for key in self.keys:
            if key != "sigma":
                keys.append(key)

        return keys


    
    def __str__(self):
        """
        Info:
            Print out infomation

        """
        pstr = ""

        pstr += "##======== Best fit parameters =========##\n"
        pstr += "##======================================##\n"

        for key in self.keys:
            if key != "sigma": 
                pstr += f"{key}:    {self.val[key]:.4f} (+{self.plus[key]:.4f}/-{self.minus[key]:.4f})\n"
        
        return pstr
            
        






##===============================================##
##             utility functions                 ##
##===============================================##

def print_bestfit(p: dict = None):
    """
    Info:
        Print out bestfit 

    Args:
        p (dict): Posterior from Fit


    """

    print("##======== Best fit parameters =========##")
    print("##======================================##")

    for key in p.keys():
        if key != "sigma": 
            print(f"{key}:    {p[key].val:.4f} (+{p[key].plus:.4f}/-{p[key].minus:.4f})")




## [ GET MEDIAN AND PLUS AND MINUS VALUES FROM RESULT ] ##
def _get_posterior(result):
    """
    Info:
        Get posterior info from BILBY result class

    Args:
        result (result): BILBY result class

    Returns:
        p (dict): Dictionary of posterior params
    """

    params = {}

    # get all parameters excluding log_likelihood
    par_keys = result.posterior.columns[:-2].tolist()
    p = struct_()

    for _,key in enumerate(par_keys):
        vals = result.get_one_dimensional_median_and_error_bar(key)
        p.val = vals.median
        p.plus = vals.plus
        p.minus = vals.minus
        params[key] = deepcopy(p)


    return params




def _proc_priors(priors: dict = None, keys: list = None):

    """
    Info:
        process priors, mainly initialise priors with any missing keys
    
    Args:
        priors (dict): Dictionary of priors in BILBY format
        keys (dict): Keys to compare against

    Returns:
        priors (dict): Updated Dictionary of priors in BILBY format

    
    """

    new_priors = {}
    prior_keys = priors.keys()
    unknown_keys = []

    for key in keys:
        if key not in prior_keys:
            unknown_keys.append(key)
            # give default value
            new_priors[key] = [0.0, 1.0]
        else:
            new_priors[key] = priors[key]


    # print info on priors
    log("##======= Priors =======##", lpf = False)
    [log(f"{key}:     {new_priors[key]}", lpf = False) for key in keys]

    if len(unknown_keys) > 0:
        log(f"NOTE: The following priors have been set to the default value of {[0.0, 1.0]}", stype = "warn",
            lpf = False)
        log(f"{unknown_keys}", lpf = False)
    
    return new_priors


def _check_static_priors(static_priors):
    if len(static_priors) > 0:
        log("NOTE: The following parameters will remain static", stype = "warn", lpf = False)
        [log(f"{skey}: {static_priors[skey]}", lpf = False) for skey in static_priors.keys()]






##===============================================##
##      additional Bilby extentions              ##
##===============================================##


## [ UNIFORM PRIORS ] ##
def priorUniform(p):
    """
    Info:
        Convert priors to prior UNIFORM usable BILBY format
    """
    priors = {}
    for _,key in enumerate(p.keys()):
        priors[key] = bilby.core.prior.Uniform(p[key][0],p[key][1],key)
    
    return priors









##======================================##
## WRAPPER FUNCTIONS FOR BETTER UTILITY ##
##======================================##

def _static_wrap_fit_func(method: str, fit_priors: dict = None, static_priors: dict = None):

    """
    Info:
        Create wrapper for function using static priors.

        Example: method = lambda x, a, b, c : _func(x, a, b, c)
        wrap:    lambda x, a, b : method(x, a, b, c = c)

    Args:
        method (str): Fitting function to wrap
        fit_priors (dict): Fit parameters
        static_priors (dict): static priors

    Returns:
        func (lambda): return wrapper function

    """
    
    # keys
    static_keys = static_priors.keys()

    # special edge case
    if method == "scatt_pulse_profile":

        # this is a unique case where the wrapped function requries a dictionary
        # of priors
        # TODO: If I need to do this for another function again, consider
        # creating a seperate wrapping function for dictionary set priors
        args_str = "lambda x"
        func_str = "scatt_pulse_profile(x,{"

        # loop through all parameters
        for key in fit_priors.keys():
            if key in static_keys:
                func_str += f"'{key}':{static_priors[key]},"

            else:
                func_str += f"'{key}':{key},"
                args_str += f",{key}"

        # put all together
        evalstr = args_str + " : " + func_str[:-1] + "})"


    # all other functions whose parameters are defined explicitly
    else:


        # get ordered arguments
        args = list(inspect.signature(eval(method)).parameters.keys())

        # start building lambda function
        args_str = "lambda " + args[0]
        func_str = method + "(" + args[0]

        for key in args[1:]:
            # check if key is static
            if key in static_keys:
                func_str += f",{key} = {static_priors[key]}"

            else:
                args_str += f",{key}"
                func_str += f",{key} = {key}"

        # put all together
        evalstr = args_str + " : " + func_str + ")"
        print(evalstr)

    return eval(evalstr)






    
# TODO: tidy up code here
## [ FIT FOR N-PULSES AND SCATTERING TAIL ] ##
def func_fit(x, y, fit_func, fit_priors: dict = None, static_priors: dict = None, **kwargs):
    """
    Info:
        Fit function using Baysian inference (Bilby)

    Args: 
        x (ndarray): x data
        y (ndarray): y data
        fit_priors (dict): initial fitting parameters
        static_priors (dict): priors to keep constant
        **kwargs: fitting parameters for BILBY

    Returns:
        params (fit_par): fit parameter class container
    
    """

    ##=======================================##
    ##  seperate variable and static priors  ##
    ##=======================================##


    # get ALL variable priors (remove static priors)
    all_priors = deepcopy(fit_priors)           

    # seperate sigma
    sigma = all_priors['sigma']
    del all_priors['sigma']         

    # put fit priors in seperate dict
    fit_priors = {}                             
    static_keys = static_priors.keys()

    # now actually do the seperating
    for key in all_priors.keys():
        if key not in static_keys:
            fit_priors[key] = all_priors[key]



    ##=======================================##
    ##        Build function to sample       ##
    ##=======================================##

    FUNC = _static_wrap_fit_func(fit_func, all_priors, static_priors)


    ##=======================================##
    ##       fit function using BILBY        ##
    ##=======================================##


    # process sigma (noise), if static add to likelihood estimator/curve_fit, else process as
    # a fitted parameter (only if method == "bays")
    fit_sigma = None

    # process sigma
    if "sigma" in static_keys:
        fit_sigma = static_priors["sigma"]
    else:
        fit_priors['sigma'] = sigma
    
    # uniform prior sampling
    fit_priors = priorUniform(fit_priors)

    # make likelihood (gaussian)
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, FUNC, sigma = fit_sigma)

    result_ = bilby.run_sampler(likelihood = likelihood, priors = fit_priors, 
                                sampler = "dynesty",**kwargs)
    

    #plot
    result_.plot_with_data(FUNC,x,y)
    result_.plot_corner()

    p = fit_par()

    # extract posteriors
    p.result2par(result_)

    # add static params to posterior struct
    p.static2par(static_priors)


    return p







##===============================================##
##           basic fitting functions             ##
##===============================================##

## [ LORENTZIAN FUNCTION ] ##
def lorentz(x,w,a):
    """
    Info:
        Lorentz function

    Args:
        x (ndarray): x data array
        w (float): HWHM 
        a (float): Amplitude

    Returns:
        y (ndarray): y data array
    """

    return a*w**2/(w**2+x**2)


## [ GAUSSIAN FUNCTION ] ##
def gaussian(x, a, mu, sig):
    """
    Info:
        Gaussian function

    Args:
        x (ndarray): x data array
        a (float): Amplitude
        mu (float): Position
        sig (float): guassian width

    Returns:
        y (ndarray): y data array
    """

    return a*np.exp(-(x-mu)**2/(2*sig**2))


def scat(x, tau):
    """
    Info:
        Scattering tail function

    Args:
        x (ndarray): x data array
        tau (float): scattering timescale 

    Returns:
        y (ndarray): y data array
    """

    # create x with same time resolution
    dt = x[1] - x[0]
    hw = x.size//2
    yscat = np.zeros(x.size)

    # only fill one side, since this is a one sided
    # exponential
    yscat[x.size-hw:] = np.exp(-np.linspace(0,dt*(hw-1),hw)/(tau))

    return yscat


def specindex(x, a, alpha):
    """
    Info:
        Spectral index function

    Args:
        x (ndarray): x data array
        a (float): Amplitude
        alpha (float): Spectral index

    Returns:
        y (ndarray): y data array
    """

    # spectral index function
    return a*x**alpha






##===============================================##
##          Advanced fitting functions           ##
##===============================================##
# TODO: Make so the scat pulse is a small as possible, maybe also implement deconvolution
def scatt_pulse_profile(x, p):
    """
    Info:
        Scattering time series profile with n pulses

    Args:
        x (ndarray): x data array
        p (dict): dictionary of params for pulses, for each pulse n:
                  [a[n]] -> Pulse amplitude
                  [mu[n]] -> Pulse position
                  [sig[n]] -> Pulse width
                  [tau] -> scattering timescale

    Returns:
        y (ndarray): y data array
    """

    npulses = (len(p) - 1)//3


    #combined guassians
    y = np.zeros(x.size)
    conv = np.zeros(x.size)
    stail = scat(x,p['tau'])

    for i in range(npulses):
        
        conv = np.convolve(gaussian(x,1,p['mu'+str(i+1)],p['sig'+str(i+1)]),
                            stail,mode = "same")

        y += p['a'+str(i+1)] * conv/np.max(conv)

    return y







##============================##
##          fitting           ##
##============================##



## RM fitting functions ##
def fit_RMsynth(I, Q, U, Ierr, Qerr, Uerr, f, clean_cutoff = 0.1, **kwargs):
    """
    Info:
        Use RM synthesis to calculate RM, pa0 and f0,
        f0 is the weighted midband frequency and pa0 the
        pa at f0.

    Args:
        I (ndarray): stokes I spectra
        Q (ndarray): stokes Q spectra
        U (ndarray): stokes U spectra
        Ierr (ndarray): stokes I rms spectra
        Qerr (ndarray): stokes Q rms spectra
        Uerr (ndarray): stokes U rms spectra
        f (ndarray): frequencies [MHz]
        clean_cutoff (float): cutoff arg for run_rmclean()
        **kwargs: keyword arguments for RM tools run_synthesis

    Returns:
        rm (float): rotation measure
        rm_err (float): error in rotation measure
        f0 (float): reference frequency at weighted mid-band
        pa0 (float): position angle at f0 

    
    """
    defkwargs = {"polyOrd":3, "phiMax_radm2":1.0e3, "dPhi_radm2":1.0, "nSamples":100.0}

    ## process kwargs keys
    keys = kwargs.keys()
    for key in defkwargs.keys():
        if key not in keys:
            kwargs[key] = defkwargs[key]


    log("Fitting RM using RM synthesis", lpf = False)
    # RM data array
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

    # run RM synthesis
    rm_sum, rm_data = run_rmsynth(rmsyn_data, **kwargs)

    # apply RM cleaning
    rmc = run_rmclean(rm_sum, rm_data, clean_cutoff)

    # get estimated parameters
    rm = rmc[0]['phiPeakPIfit_rm2']                                             # RM
    rm_err = rmc[0]['dPhiPeakPIfit_rm2']                                        # RM err
    f0 = rm_sum['freq0_Hz'] / 1e6                                               # f0 (MHz)
    pa0 = 0.5 * np.arctan2(rmc[0]['peakFDFimagFit'],rmc[0]['peakFDFrealFit'])   # pa0 (at f0)

    # print
    log(f"RM: {rm:.4f}  +/-  {rm_err:.4f}     (rad/m2)", lpf = False)
    log(f"f0: {f0}    (MHz)", lpf = False)
    log(f"pa0:  {pa0}     (rad)", lpf = False)

    return rm, rm_err, f0, pa0














def fit_RMquad(Q, U, Qerr, Uerr, f, f0, **kwargs):
    """
    Info:
        Use Quadratic method to fit for RM and pa0.

    Args:
        Q (ndarray): stokes Q or Q/I spectra
        U (ndarray): stokes U or U/I spectra
        f (ndarray): frequency array [MHz]
        f0 (float): reference frequency
        **kwargs: kwargs for scipy.curve_fit()

    Returns:
        rm (float): rotation measure
        rm_err (float): error in rotation measure
        pa0 (float): position angle at f0
        pa0_err (float): error in position angle at f0

    """
    log("Fitting using RM quadratic function", lpf = False)
    # fit RM using Quadratic function
    def rmquad(f, rm, pa0):
        return pa0 + rm*c**2/1e12*(1/f**2 - 1/f0**2)
    

    PA_meas, PA_err = calc_PA(Q, U, Qerr, Uerr)
    PA_meas = np.unwrap(PA_meas, period = np.pi)

    # fit
    fit_val, fit_err = curve_fit(rmquad, f, PA_meas, sigma = PA_err, absolute_sigma = True, **kwargs)
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

    return rm, rm_err, pa0, pa0_err





    







def fit_tscatt(t, I, npulse, priors, static_priors, **kwargs):
    """
    Info:
        Fit time series with some x number of convolved
        gaussians

    Args:
        t (ndarray): Time array in [ms]
        I (ndarray): Stokes I time series
        npulse (int): Number of convolved gaussians to fit
        priors (dict): priors, for each pulse n:
                       [a[n]] -> amplitude
                       [mu[n]] -> Position in [ms]
                       [sig[n]] -> HWHM in [ms]
                       then
                       [tau] -> Scattering time scale
        static_priors (dict): priors to keep static during fitting
                              and their vales
                              [sigma] -> if sigma is static, an array can be 
                                         used of the same size as t
        **kwargs: Additional arguments for BILBY [run_sampler] function

    Return:
        p (fit_par): Posterior data container
    
    """
    log("Fitting for scattering time scale", lpf = False)

    # process priors, if unspecified priors are present, give default values
    keys = ["tau", "sigma"]
    for i in range(1,npulse+1):
        keys.extend([f"a{i}", f"mu{i}", f"sig{i}"])
    priors = _proc_priors(priors, keys)
        
    # check if any parameters have been specified as static
    _check_static_priors(static_priors)


    ##==================##
    ##   Do Fitting     ##
    ##==================##

    p = func_fit(x = t, y = I,fit_func = "scatt_pulse_profile",
                    fit_priors = priors, static_priors = static_priors, 
                    **kwargs)

    return p





def fit_scint(f, I, priors, static_priors, **kwargs):
    """
    Info:
        Fit sectra for scintillation

    Args:
        f (ndarray): Frequency array in MHz [in asending order]
        I (ndarray): Stokes I spectra
        priors (dict): priors
                       [a] -> amplitude of lorentz function 
                       [w] -> Scintillation bandwidth
                       [sigma] -> rms in freq
        static_priors (dict): priors to keep static during fitting
                              and their vales
                              [sigma] -> if sigma is kept static, an array can
                              be used with the same size as f
        **kwargs: Additional arguments for BILBY [run_sampler] function

    Return:
        p (fit_par): Posterior data container
    
    """
    log("Fitting for scintillation bandwidth", lpf = False)

    # proc priors + static priors
    priors = _proc_priors(priors)
    _check_static_priors(static_priors)


    ## get I residuals and acf
    res = residuals(I)
    acf = acf(res)

    ## get freq lags
    df = f[1] - f[0]
    bw = f[-1] - f[0]
    nchan = acf.size
    lags = np.linspace(df, bw - df, nchan)


    ##==================##
    ##   do fitting     ##
    ##==================##

    p = func_fit(x = lags, y = acf, fit_func = "lorentz",
                    fit_priors = priors, static_priors = static_priors,
                    **kwargs) 

    return p














##===============================================##
##                 calculations                  ##
##===============================================##

# def calc_scint_C(v, t):

#     return 2 * 3.1415926 * v * t