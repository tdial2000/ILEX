##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 25/09/2023 
##
##
## 
## 
## Library of functions for HTR processing (coherent)
## 
## 
##
##===============================================##
##===============================================##
# imports
import numpy as np
from scipy.fft import fft, ifft, next_fast_len
import sys, os
from .data import rotate_data, average
from math import ceil
from .globals import *



##=====================##
## AUXILIARY FUNCTIONS ##
##=====================##

def phasor_DM(f, DM: float, f0: float):
    """
    Calculate Phasor Rotator for DM dispersion

    Parameters
    ----------
    f : np.ndarray
        Frequency array [MHz]
    DM : float
        Dispersion Measure [pc/cm^3]
    f0 : float
        Reference Frequency [MHz]

    Returns
    -------
    phasor_DM : np.ndarray
        Phasor Rotator array in frequency domain
    """
    # constants
    kDM = 4.14938e3         # DM constant

    return np.exp(2j*np.pi*kDM*DM*(f-f0)**2/(f*f0**2)*1e6)



def phasor_(f, tau: float, phi: float):
    """
    Calculate General Phasor Rotator

    Parameters
    ----------
    f : np.ndarray
        Frequency array [MHz]
    tau : float
        Time delay (us)
    phi : float
        phase delay (Rad)

    Returns
    -------
    phasor : np.ndarray
        Phasor Rotator array in frequency domain
    """
    return np.exp(2j*np.pi*tau*f + 1j*phi)
    




##==================##
## STOKES FUNCTIONS ##
##==================##

def stk_I(X, Y):
    """
    Claculate Stokes I from X and Y polarisations

    Parameters
    ----------
    X : np.ndarray
        X polarisation data
    Y : np.ndarray
        Y polarisation data

    Returns
    -------
    I : np.ndarray
        Stokes I data
    """

    return np.abs(X)**2 + np.abs(Y)**2



def stk_Q(X, Y):
    """
    Claculate Stokes Q from X and Y polarisations.

    Parameters
    ----------
    X : np.ndarray
        X polarisation data
    Y : np.ndarray
        Y polarisation data

    Returns
    -------
    Q : np.ndarray
        Stokes Q data
    """

    return np.abs(Y)**2 - np.abs(X)**2

def stk_U(X, Y):
    """
    Claculate Stokes U from X and Y polarisations

    Parameters
    ----------
    X : np.ndarray
        X polarisation data
    Y : np.ndarray
        Y polarisation data

    Returns
    -------
    U : np.ndarray
        Stokes U data
    """

    return 2 * np.real(np.conj(X) * Y)

def stk_V(X, Y):
    """
    Claculate Stokes V from X and Y polarisations

    Parameters
    ----------
    X : np.ndarray
        X polarisation data
    Y : np.ndarray
        Y polarisation data

    Returns
    -------
    V : np.ndarray
        Stokes V data
    """

    return 2 * np.imag(np.conj(X) * Y)

## array of stokes functions ##
Stk_Func = {"I":stk_I, "Q":stk_Q, "U":stk_U, "V":stk_V}

