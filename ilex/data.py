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
    average in either frequency or time

    Parameters
    ----------
    x: ndarray
       data to average over
    axis: int 
          axis to average over
    weights: ndarray
          weights to apply to data when summing over. By default this is set
          to None, in which case a uniform weighting is applied.

    Returns
    -------
    x: ndarray 
       Average data
    
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
    Scrunch data along a given axis, weights may also be applied.

    Parameters
    ----------
    x: np.ndarray
        data to scrunch
    axis: int, optional
        axes to scrunch over, by default 0
    weights: array-like or float, optional
        weights to apply during scrunching, by default None

    Returns
    -------
    y: np.ndarray
        Weighted and scrunched data
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
    Slice 1D or 2D data in phase, between 0.0-1.0 which represents the 
    start and end of ndarray along given axis. If array is 1D, given axis
    is set to 0.

    Parameters
    ----------
    x : np.ndarray
        1D or 2D data array 
    start : float
        starting point of slice 
    end : float
        ending point of slice
    axis : int, optional
        axis to slice along, can be 0 or 1(-1)

    Returns
    -------
    y: np.ndarray
        sliced 1D or 2D data array
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
    Rotate data in 2D

    Parameters
    ----------
    A : np.ndarray
        First array
    B : np.ndarray
        Second array
    angle : float
        angle [rad] to rotate A and B in 2D

    Returns
    -------
    X: np.ndarray
        First array rotated
    Y: np.ndarray
        Second array rotated
    """    

    # apply rotation between A and B (and errors)
    X = A*np.cos(angle) - B*np.sin(angle)
    Y = A*np.sin(angle) + B*np.cos(angle)

    return X, Y

    



def f_weight(x, fW):
    """
    Apply frequency weights on a 2D array, it is assumed that
    the freq axis is axis = 0

    Parameters
    ----------
    x : np.ndarray
        2D array, (f, t)
    fW : np.ndarray or float
        frequency weights

    Returns
    -------
    np.ndarray
        weighted 2D array
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
    Apply time weights on a 2D array, it is assumed that
    the time axis is axis = 1

    Parameters
    ----------
    x : np.ndarray
        2D array, (f, t)
    fW : np.ndarray or float
        time weights weights

    Returns
    -------
    np.ndarray
        weighted 2D array
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
    Normalise data

    Parameters
    ----------
    x : np.ndarray
        data to normalise
    method : str, optional
        method of normalising, by default "abs_max" \n
        [abs_max] - Normalise using absolute maximum abs(max) \n
        [max] - Normalise using maximum \n
        [unit] - normalise data between -1 and 1 - not implemented 

    Returns
    -------
    np.ndarray
        normalised data
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
    Apply Faraday rotation to 1D or 2D Stokes data

    Parameters
    ----------
    Q : np.ndarray
        Stokes Q data
    U : np.ndarray
        Stokes U data
    f : np.ndarray
        frequency array
    RM : float
        Rotation Measure [rad/m2]
    f0 : float
        reference frequency [MHz]
    pa0 : float, optional
        reference position angle [rad], by default 0.0

    Returns
    -------
    Qrot : np.ndarray
        de-faraday rotated Stokes Q
    Urot : np.ndarray
        de-faraday rotated Stokes U 
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







# def denoise():
#     pass




# def zap_chan():
#     pass








##===============================================##
##         Aditional Stokes params calcs         ##
##===============================================##


def calc_PA(Q, U, Qerr, Uerr):
    """
    Calculate Position Angle (PA) and PA angle

    Parameters
    ----------
    Q : np.ndarray
        Stokes Q data
    U : np.ndarray
        Stokes U data
    Qerr : np.ndarray
        Stokes Q err data
    Uerr : np.ndarray
        Stokes U err data

    Returns
    -------
    PA : np.ndarray
        Position Angle (PA)
    PAerr : np.ndarray
        Position Angle err  
    """

    # calculate PA and error
    PA = 0.5 * np.arctan2(U, Q)
    PAerr = 0.5 * np.sqrt((Q**2*Uerr**2 + Q**2*Qerr**2)/
                           (Q**2 + Q**2)**2)

    return PA, PAerr








def calc_PAdebiased(stk, Ldebias_threshold = 2.0):
    """
    Calculate time-dependant Position Angle masked using
    stokes L debiased

    Parameters
    ----------
    stk : Dict(np.ndarray)
        Dictionary of Stokes data \n
        [dsQ] - Stokes Q dynamic spectra \n
        [dsU] - Stokes U dynamic spectra \n
        [tQerr] - Stokes Q average rms over time \n
        [tUerr] - Stokes U average rms over time \n
        [tIerr] - Stokes I average rms over time \n
        [fQerr] - Stokes Q frequency dependent rms \n
        [fUerr] - Stokes U frequency dependent rms 
    Ldebias_threshold : float, optional
        sigma threshold for masking PA, by default 2.0

    Returns
    -------
    PA : np.ndarray
        Position Angle (PA)
    PAerr : np.ndarray
        Position Angle err 
    """    
    
    # calculate time series and de-noise
    stk_den = {}    # denoised stk data
    for S in "QU":
        stk_den[f't{S}'] = np.mean(stk[f"ds{S}"] - stk[f"f{S}err"][:, None], axis = 0)

    # calculate PA and error
    PA = 0.5 * np.arctan2(stk_den['tU'], stk_den['tQ'])
    PAerr = 0.5 * np.sqrt((stk_den['tQ']**2*stk['tUerr']**2 + stk_den['tU']**2*stk['tQerr']**2)/
                           (stk_den['tQ']**2 + stk_den['tU']**2)**2)

    # calculate de-baised L and mask PA 
    L_debias = calc_Ldebiased(stk_den['tQ'], stk_den['tU'], stk['tIerr'])
    PA_mask = L_debias < Ldebias_threshold * stk['tIerr']

    # mask PA
    PA[PA_mask] = np.nan
    PAerr[PA_mask] = np.nan

    return PA, PAerr





def calc_L(Q, U):
    """
    Calculate Stokes L

    Parameters
    ----------
    Q : np.ndarray
        Stokes Q data
    U : np.ndarray
        Stokes U data

    Returns
    -------
    L : np.ndarray
        Stokes L data
    """

    return np.sqrt(Q**2 + U**2)






def calc_Ldebiased(Q, U, Ierr):
    """
    Calculate De-biased Linear polarisation fraction
    (see Everett & Weisberg+2001)

    Parameters
    ----------
    Q : np.ndarray
        Stokes Q data
    U : np.ndarray
        Stokes U data
    Ierr : np.ndarray
        Stokes U err data

    Returns
    -------
    L_debias : np.ndarray
        Stokes L debias
    """

    L_meas = np.sqrt(Q**2 + U**2)
    L_debias = Ierr * np.sqrt((L_meas/Ierr)**2 - 1)
    L_debias[np.isnan(L_debias)] = 0
    L_debias[L_meas/Ierr < 1.57] = 0

    return L_debias




def calc_ratio(I, X, Ierr = None, Xerr = None):
    """
    Calculate Stokes Ratio X/I 

    Parameters
    ----------
    I : np.ndarray
        Stokes I data
    X : np.ndarray
        Stokes [X] data, usually either Q, U or V
    Ierr : np.ndarray, optional
        Stokes I err data, by default None, if both Ierr and Xerr is given \n
        Stokes X/I err will also be calculated and returned
    Xerr : np.ndarray, optional
        Stokes [X] err data, by default None

    Returns
    -------
    XI : np.ndarray
        Stokes X/I data
    XIerr : np.ndarray, optional 
        Stokes X/I err data, by default None
    """
    XIerr = None

    # calc XI
    XI = X/I

    # calc error?
    if Ierr is not None and Xerr is not None:
        XIerr = np.sqrt((Xerr/I)**2 + (Ierr*X/I**2)**2)

        # check if array or scalar
        if not hasattr(Ierr, "__len__") or not hasattr(Xerr, "__len__"):
            # take standard deviation
            XIerr = np.nanmean(XIerr)

    return XI, XIerr






def calc_freqs(cfreq, bw = 336.0, df = 1.0, upper = True):
    """
    Calculate Frequencies

    Parameters
    ----------
    cfreq : float
        Central frequency [MHz]
    bw : float, optional
        Bandwidth [MHz], by default 336.0
    df : float, optional
        channel width [MHz], by default 1.0
    upper : bool, optional
        If true, freq band starts at top, by default True

    Returns
    -------
    freqs : np.ndarray
        Frequency array
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
    Calculate residuals of a data array by subtracting the mean
    model

    Parameters
    ----------
    y : np.ndarray
        data array
    n : int, optional
        max order for polynomial to fit to mean model of y, by default 5

    Returns
    -------
    np.ndarray
        residuals of y
    """

    x = np.arange(y.size)
    y_fit = np.poly1d(np.polyfit(x,y,n))
    
    return (y - y_fit(x))


## [ AUTO-CORRELATION FUNCTION ] ##
def acf(x, outs = "unique"):
    """
    Calculate normalised Auto-Correlation function using 'fft' method of 
    real-valued data

    Parameters
    ----------
    x : np.ndarray
        data array
    outs : str, optional
        describes output of acf function, by default "unique" \n
        [unique] - Take positive acf lags, exclude zero-lag peak
        [all] - Take full acf, including positive, negative and zero lags

    Returns
    -------
    np.ndarray
        Auto-Correlation of x data
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
    """
    Calculate normalised Cross-Correlation function using 'fft' method of 
    real-valued data

    Parameters
    ----------
    x : np.ndarray
        data array
    outs : str, optional
        describes output of ccf function, by default "unique" \n
        [unique] - Take positive acf lags, including zero-lag peak
        [all] - Take full acf, including positive, negative and zero lags

    Returns
    -------
    np.ndarray
        Cross-Correlation of x data
    """

    #correlate
    xcorr = correlate(x,y,mode = "full",method = "fft")/(np.sum(x**2)*np.sum(y**2))**0.5


    if outs == "unique":
        return xcorr[x.size-1:]

    elif outs == "all":
        return xcorr

    else:
        return None
