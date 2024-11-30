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

from .pyfit import fit

from .globals import *



##===============================================##
##             utility functions                 ##
##===============================================##


def model_curve(y, n: int = 5, samp: int = None):
    """
    Fit Polynomial model to data

    Parameters
    ----------
    y : np.ndarray
        data to model
    n : int, optional
        Polynomial order, by default 5
    samp : int, optional
        number of samples to sample modelled data, by default None

    Returns
    -------
    np.ndarray
        Modelled data
    """    

    if samp is None:
        samp = x.size

    x = np.linspace(0, 1.0, y.size)
    xnew = np.linspace(0, 1.0, samp)
    
    y_fit = np.poly1d(np.polyfit(x,y,n))

    return y_fit(xnew)





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






##===============================================##
##          Advanced fitting functions           ##
##===============================================##
def scatt_pulse_profile(x, p):
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
# def scatt_pulse_profile(x, p):
#     """
#     Scattering time series profile with n pulses. Numerical convolution if done incorrectly
#     can shift the resultant data in an undesirable way. One way to avoid this is to take a large window
#     around the known signal to encompass the all pulses and convolve this with a symmetrical scattering tail.
#     This of course isn't realistic when taking a crop of data whose bounds cut through potential signal. 

#     To keep this function robust, the algorithm implemented here takes each gaussian profile and extends it until 
#     symmetrical, this avoids any potential shifting due to improper convolution. 

#     Parameters
#     ----------
#     x: np.ndarray
#         X data array
#     p: Dict(float)
#         dictionary of parameters for scattered Gaussian pulses, for each pulse n: \n
#         [a[n]] - Pulse amplitude \n
#         [mu[n]] - Pulse position \n
#         [sig[n]] - Pulse width \n
#         [tau] - scattering timescale

#     Returns
#     -------
#     y: np.ndarray
#         Y data array
#     """
#     # create empty output array
#     y = np.zeros(x.size)
#     print(x)
    
#     # create scattering tail 
#     dt = x[1] - x[0]
#     npulses = (len(p) - 1)//3
#     stail = scat(dt,p['tau'], sig = 3)
#     print(stail.size)
#     # plt.plot(np.linspace(0.0, 1.0, stail.size), stail)

#     # Each gaussian will be isolated and convolved seperatley with enough padding for a complete
#     # uniform convolution with zero shifting due to numerical error.
#     for i in range(npulses):

#         # make gaussian with sigma 5
#         xe = int(floor((p[f"mu{i+1}"] + p[f"sig{i+1}"]*5)/dt))  #assuming starts at zero, at increments of 
#         xs = int(floor((p[f"mu{i+1}"] - p[f"sig{i+1}"]*5)/dt))
    
#         # make sure the scattering tail is smaller or equal to the size of the gaussian to convolve
#         # if x.size < stail.size:
#         #     # expand to same size as stail
#         #     lendif = (stail.size - (x.size))
#         #     xe += lendif
#         #     xs -= lendif
            
            
#         x_i = np.linspace(xs*dt, xe*dt, xe-xs + 1)
#         print(x_i)

#         # crop bounded signal
#         ps = int(floor(x[0]/dt))
#         xs -= ps
#         xe -= ps

#         # handle edge cases
#         if xs >= x.size:
#             continue

#         if xe <= 0:
#             continue

#         # make pulse
#         pulse_i = gaussian(x, 1, p[f"mu{i+1}"], p[f"sig{i+1}"])
#         print(pulse_i.size)
#         print(x.size)
#         print(stail.size)
#         # print(pulse_i.size)
#         # plt.plot(x_i, pulse_i)

#         # convolve
#         conv = np.convolve(pulse_i, stail, mode = "same")
#         # plt.plot(np.linspace(0,1.0, stail.size),stail)
#         plt.plot(x, conv/np.max(conv)*p[f"a{i+1}"], label = f"{1/dt}")
#         pulse_ind = [0, conv.size]
        
#         if xs < 0:
#             pulse_ind[0] = 0 - xs
#             xs = 0
#         if xe + 1 > x.size:
#             pulse_ind[1] -= (xe+1 - x.size)
#             xe = x.size

#         # y[xs:xe+1] += p[f"a{i+1}"] * conv[pulse_ind[0]:pulse_ind[1]]/np.max(conv)
#     return y






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
        lambda function for scatter pulse profile with n pulses
    """

    args_str = "lambda x"
    func_str = "scatt_pulse_profile(x,{"

    for i in range(1, n+1):
        # loop through components
        for p in ["a", "mu", "sig"]:
            func_str += f"'{p}{i}':{p}{i},"
            args_str += f",{p}{i}"

    # add tau
    func_str += "'tau':tau"
    args_str += ",tau"

    return (eval(args_str + ":" + func_str + "})"))







##============================##
##          fitting           ##
##============================##



## RM fitting functions ##
def fit_RMsynth(I, Q, U, Ierr, Qerr, Uerr, f, clean_cutoff = 0.1, **kwargs):
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
    clean_cutoff: float 
        cutoff arg for run_rmclean()
    **kwargs: Dict 
        keyword arguments for RM tools run_synthesis

    Returns
    -------
    rm: float 
        rotation measure
    rm_err: float 
        error in rotation measure
    f0: float 
        reference frequency at weighted mid-band
    pa0: float 
        position angle at f0 
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
    rm: float 
        rotation measure
    rm_err: float 
        error in rotation measure
    pa0: float 
        position angle at f0 
    pa0_err: float
        position angle err
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

    return rm, rm_err, pa0, pa0_err





