##===============================================##
##===============================================##
## Author: Tyson Dial                            ##
## Email: tdial@swin.edu.au                      ##
## Last Updated: 09/01/2024                      ##
##                                               ##
##                                               ##
##                                               ##
##                                               ##
## Data processing library to manipulate Stokes  ##
## High time resolution (HTR) data               ##
##===============================================##
##===============================================##

import numpy as np
from scipy.signal import correlate
from copy import deepcopy

# constants
c = 2.997924538e8 # Speed of light [m/s]


##===============================================##
##      Basic functions to manipulate data       ##
##===============================================##

##  function to average data    ##
def average(x: np.ndarray, axis: int = 0, N: int = 10):

    """
    Info:
        average in either frequency or time

    Args:
        x (ndarray): data to average over
        axis (int): axis to average over
        weights (ndarray): weights to apply to data when summing over. By default this is set
                           to None, in which case a uniform weighting is applied.

    Returns:
        x (ndarray): Average data
    
    """

    if N == 1:
        return x
    

    # either dynamic spectra or time series
    ndims = x.ndim
    if ndims == 1:
        N_new = (x.size // N) * N
        return np.mean(x[:N_new].reshape(N_new // N, N),axis = 1).flatten()
    
    elif ndims == 2:
        if axis == 0:
            #frequency scrunching
            N_new = (x.shape[0] // N) * N
            return np.mean(x[:N_new].T.reshape(x.shape[1],N_new // N, N),axis = 2).T
        
        elif axis == 1 or axis == -1:
            #time scrunching
            N_new = (x.shape[1] // N) * N
            return np.mean(x[:,:N_new].reshape(x.shape[0],N_new // N, N),axis = 2)
        
        else:
            log("axis must be 1[-1] or 0", lpf = False)
            return x
        
    else:
        log("ndims must equal 1 or 2..", lpf = False)
        return x
    




def scrunch(x: np.ndarray, axis: int = 0, weights = None):
    """
    Info:
        collapse dynamic spectrum in either time or 
        frequency (freq -> axis = 0, time -> axis = 1)

    Args:
        x (ndarray): dynamic spectra
        axis (int): axis to collapse
        weights (ndarray): weights to apply to data when summing over. By default this is set
                           to None, in which case a uniform weighting is applied.

    Returns
        y (ndarray): 1D time/freq data

    """


    # check weights
    if weights is None:
        weights = 1.0

    if type(weights) == np.ndarray:
        if weights.size != x.shape[axis]:
            log(f"weights {weights.size} != xdata {x.size}", lpf = False)
            log("size of weights array mismatch, using uniform weighting instead", lpf = False)
            weights = 1.0

        else:
            if axis == 0:
                w_shape = (weights.size, 1)
            elif axis == 1 or axis == -1:
                w_shape = (1, weights.size)

            weights = weights.reshape(w_shape)
            



    # scrunch
    y = np.mean(x * weights, axis = axis)

    return y






##  function to index in phase  ##
def pslice(x: np.ndarray, start: float, end: float, axis: int = 0):

    """
    Info:
        Slice data in phase

    Args:
        x (ndarray): data to slice
        axis (int): axis to slice over
        start (float): starting phase
        end (float): end phase

    Returns:
        x (ndarray): sliced data 

    """

    # Check number of dims
    ndims = x.ndim
    if ndims == 1:
        # time series
        start, end = int(start*x.size), int(end*x.size)
        if start < 0 or end > x.size:
            log("Phase slicing must be between [0,1]", lpf = False)
            return None
        
        return x[start:end].copy()
    
    elif ndims == 2:
        if axis == 0:
            # frequency phase slicing
            start, end = int(start*x.shape[0]), int(end*x.shape[0])
            return x[start:end,:].copy()
        
        elif axis == 1 or axis == -1:
            start, end = int(start*x.shape[1]), int(end*x.shape[1])
            return x[:,start:end].copy()
        
        else:
            log("axis must be 1[-1] or 0", lpf = False)
            return x
    
    else:
        log("ndims must be either 1 or 2", lpf = False)
        return x





def rotate_data(A, B, angle):
    """
    Info:
        Apply rotation between 2 data products (ndarrays).

    Args:
        A (ndarray): array 1 to rotate 
        B (ndarray): array 2 to rotate
        angle (ndarray): angle to rotate A and B by

    Returns:
        X (ndarray): rotated A array
        Y (ndarray): rotated B array
        
    """

    # apply rotation between A and B (and errors)
    X = A*np.cos(angle) - B*np.sin(angle)
    Y = A*np.sin(angle) + B*np.cos(angle)

    return X, Y

    



def f_weight(x, fW):
    """
    Info:
        Apply frequency weights to dynamic spectra

    Args:
        x (ndarray): 2D dynamic spectra
        fw (int/float or 1D ndarray): frequency weights

    Returns:
        y (ndarray): frequency Weighted dynamic spectra 
    """

    if fW is not None:
        if np.isscalar(fW):
            return x * fW
        else:    
            return x * fW[:, None]
    else:
        return x




def t_weight(x, tW):
    """
    Info:
        Apply Time weights to dynamic spectra

    Args:
        x (ndarray): 2D dynamic spectra
        fw (int/float or 1D ndarray): Time weights

    Returns:
        y (ndarray): Time Weighted dynamic spectra 
    """

    if tW is not None:
        if np.isscalar(tW):
            return x * tW
        else:    
            return x * tW[None, :]
    else:
        return x





def norm(x, method = "abs_max"):
    """
    Info:
        Normalize data

    Args:
        x (ndarray): data
        method (str): method of normalisation

    Returns:
        y (ndarray): normalised data
    
    """

    # normalise using the maximum value
    if method == "max":
        x /= np.max(x)

    # normalise using the absolute maximum value
    elif method == "abs_max":
        x /= np.abs(np.max(x))

    # normalise data to between -1 and 1 [-1, 1]
    elif method == "unit":
        print("Not implemented yet")

    elif method == "None":
        pass

    else:
        print("invalid method for normalisation")


    return x













##===============================================##
##        Advanced functions for HTR data        ##
##===============================================##


def fday_rot(Q, U, f,  RM, f0, pa0 = 0.0):
    """
    Info:
        Apply Faraday Rotation via rotation about Q and U

    Args:
        Q (1D or 2D ndarray): Stokes Q dynamic spectra or spectra
        U (1D or 2D ndarray): Stokes U dynamic spectra or spectra
        f (ndarray): Frequency array
        RM (float): Rotation Measure
        f0 (float): Reference frequency

    Returns:
        Q (1D or 2D ndarray): Rotated stokes Q
        U (1D or 2D ndarray): Rotated stokes U

    """    
    if RM == 0.0 or RM is None:
        return Q, U
    if f0 is None or f0 == 0.0:
        print("Must specify non-zero f0")
        return Q, U
    
    # calculate faraday angle
    fang = RM * c**2 / 1e12 * (1/f**2 - 1/f0**2) + pa0

    # dynamic spectra or spectra?
    if Q.ndim > 1:
        fang = fang[:, None]

    # de-rotate using faraday angle
    return rotate_data(Q, U, -2*fang)







def denoise():
    pass




def zap_chan():
    pass








##===============================================##
##         Aditional Stokes params calcs         ##
##===============================================##


def calc_PA(Q, U, Qerr, Uerr):
    """
    Info:
        Calculate PA

    Args: (all inputs are de-faraday rotated for best results)
        tQ (ndarray): Stokes Q  
        tU (ndarray): Stokes U
        tQerr (ndarray): Stokes Q error 
        tUerr (ndarray): Stokes U error 

    Returns
        PA (ndarray): position angle 
        PA_err (ndarray): position angle error 

    """

    # calculate PA and error
    PA = 0.5 * np.arctan2(U, Q)
    PA_err = 0.5 * np.sqrt((Q**2*Uerr**2 + Q**2*Qerr**2)/
                           (Q**2 + Q**2)**2)

    return PA, PA_err








def calc_PAdebiased(stk, Ldebias_threshold = 2.0):
    """
    Info:
        Calculate de-biased PA as a function of time
        (see Day+2020 and references therin)

    Args:
        stk (dict): dictionary of data (Note these data products must be de-faraday rotated)
                    [Q] -> Stokes Q dynamic spectra
                    [U] -> Stokes U dynamic spectra
                    [tQerr] -> Stokes Q average rms over time 
                    [tUerr] -> Stokes U average rms over time
                    [tIerr] -> Stokes I average rms over time
                    [fQerr] -> Stokes Q frequency dependent rms
                    [fUerr] -> Stokes U frequency dependent rms
        Ldebias_threshold (float): Threshold for PA masking using L_debias

    Returns:
    PA (ndarray): Time dependant position angle
    PA_err (ndarray): Time dependant position angle error

    """

    # calculate time series and de-noise
    for S in "QU":
        stk[f't{S}'] = np.mean(stk[S] - stk[f"f{S}err"][:, None], axis = 0)

    # calculate PA and error
    PA = 0.5 * np.arctan2(stk['tU'], stk['tQ'])
    PA_err = 0.5 * np.sqrt((stk['tQ']**2*stk['tUerr']**2 + stk['tU']**2*stk['tQerr']**2)/
                           (stk['tQ']**2 + stk['tU']**2)**2)

    # calculate de-baised L and mask PA 
    L_debias = calc_Ldebiased(stk['tQ'], stk['tU'], stk['tIerr'])
    PA_mask = L_debias < Ldebias_threshold * stk['tIerr']

    # mask PA
    PA[PA_mask] = np.nan
    PA_err[PA_mask] = np.nan

    return PA, PA_err





def calc_L(Q, U):
    """
    Info:
        Calculate Linear polarisation fraction

    Args:
        Q (ndarray): Stokes Q time/frequency array
        U (ndarray): Stokes U time/frequency array

    Returns:
        L (ndarray): Stokes L time/frequency array

    """

    return np.sqrt(Q**2 + U**2)






def calc_Ldebiased(Q, U, Ierr):
    """
    Info:
        Calculate De-biased Linear polarisation fraction
        (see Everett & Weisberg+2001)

    Args:
        Q (ndarray): Stokes Q time/frequency array
        U (ndarray): Stokes U time/frequency array
        Ierr (ndarray): Stokes I time/frequency error array

    Returns:
        L (ndarray): Stokes L time/frequency array
    """

    L_meas = np.sqrt(Q**2 + U**2)
    L_debias = Ierr * np.sqrt((L_meas/Ierr)**2 - 1)
    L_debias[L_meas/Ierr < 1.57] = 0

    return L_debias




def calc_coeffs(stk_data):
    """
    Info: 
        Calculate stokes coefficients

    Args:
        stk_data (dict): Stokes data in dictionary

    Returns:
        stk_data (dict): Stokes coefficients

    """
    stk_dataout = deepcopy(stk_data)

    keys = stk_data.keys()

    if "I" not in keys:
        print("Stokes I data requried to calculate stokes coefficients")

    for S in keys:
        if S != "I":
            stk_dataout[S] /= stk_dataout["I"]
    stk_dataout['I'] /= stk_dataout['I']

    return stk_dataout






def calc_freqs(cfreq, bw = 336.0, df = 1.0, upper = True):
    """
    Calculate Frequencies
    """
    
    freqs = np.linspace(cfreq - bw/2 + df/2,
                 cfreq + bw/2 - df/2, int(bw / df))

    if upper:
        freqs = freqs[::-1]

    return freqs













##===============================================##
##      basic statistics functions               ##
##===============================================##

## [ GET RESIDUALS OF SOME FUNCTION ] ##
def residuals(y,n=5):
    """
    Info:
        Get residuals of data array

    Args:
        y (ndarray): data
        n (int): order of polynomial to fit to data

    Returns:
        res (ndarray): Residuals

    """

    x = np.arange(y.size)
    y_fit = np.poly1d(np.polyfit(x,y,n))
    
    return (y - y_fit(x))


## [ AUTO-CORRELATION FUNCTION ] ##
def acf(x, outs = "unique"):
    """
    Info:
        Calculate auto-correlation function of real-valued data

    Args:
        x (ndarray): data
        outs (str): output:
                    [unique] -> take half of acf, excluding zero-lag point

    Returns:
        acf (ndarray): acf
    """

    #correlate
    acorr = correlate(x,x,mode = "full",method = "fft")
    acorr /= np.sum(x**2)
    
    if outs == "unique":
        return acorr[x.size:]

    elif outs == "all":
        return acorr

    else:
        return None
    

## [ CROSS-CORRELATION FUNCTION ] ##
def ccf(x, y, outs = "unique"):

    #correlate
    xcorr = correlate(x,y,mode = "full",method = "fft")/(np.sum(x**2)*np.sum(y**2))**0.5


    if outs == "unique":
        return xcorr[x.size-1:]

    elif outs == "all":
        return xcorr

    else:
        return None
