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
from .globals import *


##===============================================##
##      Basic functions to manipulate data       ##
##===============================================##

##  function to average data    ##
def average(x: np.ndarray, axis: int = 0, N: int = 10, nan = False):

    """
    average in either frequency or time

    Parameters
    ----------
    x: ndarray
       data to average over
    axis: int 
        axis to average over
    N: int
        Averaging/donwsampling factor
    nan : bool, optional
        If True, using nanmean to ignore NaN values in array 'x', by default False

    Returns
    -------
    x: ndarray 
       Averaged data
    
    """

    # if nan if true, will use numpy function that ignores nans in array x
    if nan:
        func = np.nanmean
    else:
        func = np.mean


    if N == 1:
        return x
    

    # either dynamic spectra or time series
    ndims = x.ndim
    if ndims == 1:
        N_new = int(x.size / N) * N
        return func(x[:N_new].reshape(int(N_new / N), N),axis = 1).flatten()
    
    elif ndims == 2:
        if axis == 0:
            #frequency scrunching
            N_new = int(x.shape[0] / N) * N
            return func(x[:N_new].T.reshape(x.shape[1],int(N_new / N), N),axis = 2).T
        
        elif axis == 1 or axis == -1:
            #time scrunching
            N_new = int(x.shape[1] / N) * N
            return func(x[:,:N_new].reshape(x.shape[0],int(N_new / N), N),axis = 2)
        
        else:
            print("axis must be 1[-1] or 0")
            return x
        
    else:
        print("ndims must equal 1 or 2..")
        return x






def scrunch(x: np.ndarray, axis: int = 0, weights = None, nan = False):
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
    nan : bool, optional
        If True, using nanmean to ignore NaN values in array 'x', by default False

    Returns
    -------
    y: np.ndarray
        Weighted and scrunched data
    """    

    # if nan if true, will use numpy function that ignores nans in array x
    if nan:
        func = np.nanmean
    else:
        func = np.mean


    # check weights
    if weights is None:
        weights = 1.0

    if type(weights) == np.ndarray:
        if weights.size != x.shape[axis]:
            print(f"weights {weights.size} != xdata {x.size}")
            print("size of weights array mismatch, using uniform weighting instead")
            weights = 1.0

        else:
            if axis == 0:
                w_shape = (weights.size, 1)
            elif axis == 1 or axis == -1:
                w_shape = (1, weights.size)

            weights = weights.reshape(w_shape)
            



    # scrunch
    y = func(x * weights, axis = axis)

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
            print("Phase slicing must be between [0,1]")
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
            print("axis must be 1[-1] or 0")
            return x
    
    else:
        print("ndims must be either 1 or 2")
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





def norm(x, method = "abs_max", nan = False):
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
    nan : bool, optional
        If True, using nanmax to ignore NaN values in array 'x', by default False

    Returns
    -------
    np.ndarray
        normalised data
    """

    # if nan if true, will use numpy function that ignores nans in array x
    if nan:
        func = np.nanmax
    else:
        func = np.max

    # normalise using the maximum value
    if method == "max":
        x /= func(x)

    # normalise using the absolute maximum value
    elif method == "abs_max":
        x /= np.abs(func(x))

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




# create a function to zap channels
def zap_chan(f, zap_str):
    """
    Zap channels, assumes contiguous frequency array

    Parameters
    ----------
    f : np.ndarray
        frequency array used for zapping
    zap_str : str
        string used for zapping channels, in format -> "850, 860, 870:900" \n
        each element seperated by a ',' is a seperate channel. If ':' is used, user can specify a range of values \n
        i.e. 870:900 -> from channel 870 to 900 inclusive of both.

    Returns
    -------
    y : np.ndarray
        zapped indicies in frequency
    
    """

    # vals
    df = f[1] - f[0]
    f_min = np.min(f)
    f_max = np.max(f)

    if df < 0:
        # upperside band
        fi = f_max
        df_step = -1

    else:
        # lowerside band
        fi = f_min
        df_step = 1

    df = abs(df)
    

    # split segments
    zap_segments = zap_str.split(',')
    seg_idx = []

    # for each segment, check for delimiter :, else float cast
    for i, zap_seg in enumerate(zap_segments):

        # if segment is a range of frequencies
        if ":" in zap_seg:
            zap_range = zap_seg.strip().split(':')
            zap_0 = round(df_step * (float(zap_range[0]) - fi)/df)
            zap_1 = round(df_step * (float(zap_range[1]) - fi)/df)

            # check if completely outside bounds
            if (zap_0 < 0 and zap_1 < 0) or (zap_0 > f.size -1 and zap_1 > f.size -1):
                print(f"zap range [{zap_range[0]}, {zap_range[1]}] MHz out of range of bandwidth [{f_min}, {f_max}] MHz")
                continue            
            
            # check bounds
            crop_zap = False

            if zap_0 < 0:
                crop_zap = True
                zap_0 = 0
            elif zap_0 > f.size - 1:
                crop_zap = True
                zap_0 = f.size - 1

            if zap_1 < 0:
                crop_zap = True
                zap_1 = 0
            elif zap_1 > f.size - 1:
                crop_zap = True
                zap_1 = f.size - 1

            if crop_zap:
                print(f"zap range cropped from [{zap_range[0]}, {zap_range[1]}] MHz -> [{f[zap_0]}, {f[zap_1]}] MHz")

            seg_idx += list(range(zap_0,zap_1+1,df_step))[::df_step]

        
        # if segment is just a single frequency
        else:
            _idx = round(df_step * (float(zap_seg.strip()) - fi)/df)
            if (_idx < 0) or (_idx > f.size - 1):
                print(f"zap channel {zap_seg.strip()} MHz out of bounds of bandwidth [{f_min}, {f_max}] MHz")
            else:
                seg_idx += [_idx]

    return seg_idx






def get_zapstr(chan, freq):
    """
    Create string of channels to zap based on given nan frequencies in 
    stokes I dynamic spectrum

    Parameters
    ----------
    chan : np.ndarray or array-like
        Stokes I freq array
    freq : np.ndarray or array-like
        Array of frequency values in [MHz]

    Returns
    -------
    zap_str: str
        string of frequencies to zap using zap_chan function
    
    """

    # could be improved later with a smarter algorithm, but not nessesary for ilex.

    zap_str = ""

    chan2zap = np.argwhere(np.isnan(chan)).flatten()
    
    i = 0
    while i < chan2zap.size:
        j = 0
        while i + j + 1 < chan2zap.size:
            if chan2zap[i + 1 + j] - chan2zap[i + j] == 1:
                j += 1
            else:
                break

        if j > 3:
            zap_str += "," + str(freq[chan2zap[i]]) + ":" + str(freq[chan2zap[i + j]])
        else:
            for k in range(j+1):
                zap_str += f",{freq[chan2zap[i+k]]}"
        
        i += j + 1

        
    if zap_str != "":
        zap_str = zap_str[1:]   # remove ','
    
    return zap_str





##===============================================##
##         Aditional Stokes params calcs         ##
##===============================================##


def calc_PA(Q, U, Qerr, Uerr, rad2deg = False):
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
    rad2deg: bool, optional
        If true, converts output to degrees, by default is False

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
    
    # convert to degrees if requested
    if rad2deg:
        PA *= 180/np.pi
        PAerr *= 180/np.pi


    return PA, PAerr








def calc_PAdebiased(stk, Ldebias_threshold = 2.0, rad2deg = False):
    """
    Calculate time-dependant Position Angle masked using
    stokes L debiased

    Parameters
    ----------
    stk : Dict(np.ndarray)
        Dictionary of Stokes data \n
        [tQ] - Stokes Q dynamic spectra \n
        [tU] - Stokes U dynamic spectra \n
        [tQerr] - Stokes Q average rms over time \n
        [tUerr] - Stokes U average rms over time \n
        [tIerr] - Stokes I average rms over time 
    Ldebias_threshold : float, optional
        sigma threshold for masking PA, by default 2.0
    rad2deg: bool, optional
        If true, converts output to degrees, by default is False

    Returns
    -------
    PA : np.ndarray
        Position Angle (PA)
    PAerr : np.ndarray
        Position Angle err 
    """    
    # calculate PA and error
    PA = 0.5 * np.arctan2(stk['tU'], stk['tQ'])
    PAerr = 0.5 * np.sqrt((stk['tQ']**2*stk['tUerr']**2 + stk['tU']**2*stk['tQerr']**2)/
                           (stk['tQ']**2 + stk['tU']**2)**2)

    # calculate de-baised L and mask PA 
    L_debias,_ = calc_Ldebiased(stk['tQ'], stk['tU'], stk['tIerr'])
    PA_mask = L_debias < (Ldebias_threshold * stk['tIerr'])

    # mask PA
    PA[PA_mask] = np.nan
    PAerr[PA_mask] = np.nan

    if rad2deg:
        PA *= 180/np.pi
        PAerr *= 180/np.pi

    return PA, PAerr





def calc_L(Q, U, Qerr = None, Uerr = None):
    """
    Calculate L

    Parameters
    ----------
    Q : np.ndarray
        Stokes Q data
    U : np.ndarray
        Stokes U data
    Qerr : np.ndarray
        Stokes Q error
    Uerr : np.ndarray
        Stokes U error

    Returns
    -------
    L : np.ndarray
        L data
    Lerr : np.ndarray
        L errors
    """

    # calc L
    L = np.sqrt(Q**2 + U**2)

    # calc Error in L
    Lerr = None
    if (Qerr is not None) and (Uerr is not None):
        Lerr = np.sqrt(Q**2*Qerr**2 + U**2*Uerr**2)
        Lmask = L != 0.0
        Lerr[Lmask] = Lerr[Lmask]/L[Lmask]
        Lerr[~Lmask] = np.nan
        L[~Lmask] = np.nan

    return L, Lerr










def calc_Ldebiased(Q, U, Ierr, Qerr = None, Uerr = None):
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
    Qerr : np.ndarray
        Stokes Q error
    Uerr : np.ndarray
        Stokes U error

    Returns
    -------
    L_debias : np.ndarray
        Stokes L debias
    Lerr : np.ndarray
        L errors
    """

    L_meas = np.sqrt(Q**2 + U**2)
    L_debias = Ierr * np.sqrt((L_meas/Ierr)**2 - 1)
    L_debias[np.isnan(L_debias)] = 0
    L_debias[L_meas/Ierr < 1.57] = 0

    Lerr = None
    if (Qerr is not None) and (Uerr is not None):
        Lerr = np.sqrt(Q**2*Qerr**2 + U**2*Uerr**2)
        Lmask = L_debias != 0.0
        Lerr[Lmask] = Lerr[Lmask]/L_debias[Lmask]
        Lerr[~Lmask] = np.nan
        L_debias[~Lmask] = np.nan

    return L_debias, Lerr








def calc_P(Q, U, V, Qerr = None, Uerr = None, Verr = None):
    """
    Calculate P

    Parameters
    ----------
    Q : np.ndarray
        Stokes Q data
    U : np.ndarray
        Stokes U data
    V : np.ndarray
        Stokes V data
    Qerr : np.ndarray
        Stokes Q error
    Uerr : np.ndarray
        Stokes U error
    Verr : np.ndarray
        Stokes V error

    Returns
    -------
    P : np.ndarray
        P data
    Perr : np.ndarray
        P errors
    """

    # calculate L
    L, Lerr = calc_L(Q, U, Qerr, Uerr)
    # calculate P
    P = np.sqrt(L**2 + V**2)

    # calculate Perr
    Perr = None
    if (Lerr is not None) and (Verr is not None):
        Perr = np.sqrt(L**2*Lerr**2 + V**2*Verr**2)
        Pmask = P != 0.0
        Perr[Pmask] = Perr[Pmask]/P[Pmask]
        Perr[~Pmask] = np.nan
        P[~Pmask] = np.nan

    return P, Perr








def calc_Pdebiased(Q, U, V, Ierr, Qerr = None, Uerr = None, Verr = None):
    """
    Calculate De-biased Linear polarisation fraction then calculate
    total polarisation (see Everett & Weisberg+2001)

    Parameters
    ----------
    Q : np.ndarray
        Stokes Q data
    U : np.ndarray
        Stokes U data
    V : np.ndarray
        Stokes V data
    Ierr : np.ndarray
        Stokes U err data
    Qerr : np.ndarray
        Stokes Q error
    Uerr : np.ndarray
        Stokes U error
    Verr : np.ndarray
        Stokes V error

    Returns
    -------
    P_debias : np.ndarray
        P debias
    Perr : np.ndarray
        P errors
    """

    # calc L debais
    L_debias, Lerr = calc_Ldebiased(Q, U, Ierr, Qerr, Uerr)
    # calc P 
    P_debias = np.sqrt(L_debias**2 + V**2)

    # calc P err
    Perr = None
    if (Lerr is not None) and (Verr is not None):
        Perr = np.sqrt(L_debias**2*Lerr**2 + V**2*Verr**2)
        Pmask = P_debias != 0.0
        Perr[Pmask] = Perr[Pmask]/P_debias[Pmask]
        Perr[~Pmask] = np.nan
        P_debias[~Pmask] = np.nan

    return P_debias, Perr







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
    if (Ierr is not None) and (Xerr is not None):
        XIerr = np.sqrt((Xerr/I)**2 + (Ierr*X/I**2)**2)

        # # check if array or scalar
        # if keep_size:
        #     if not hasattr(Ierr, "__len__") or not hasattr(Xerr, "__len__"):
        #         # take standard deviation
        #         XIerr = np.nanmean(XIerr)

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
    model. This function can handle NaN values

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

    # set x vals to nans for corrosponding y values
    mask = np.isnan(y)

    y_fit = np.poly1d(np.polyfit(x[~mask],y[~mask],n))
    y_out = y.copy()
    y_out[~mask] -= y_fit(x[~mask])
    
    return y_out, y_fit




def _nanacf(x):
    """
    autocorrelate with NaN values

    Parameters
    ----------
    x : np.ndarray (1D) 
        1D vector to correlate
    
    """

    kmax = 2*x.size - 1

    # output vector of 1D correlation
    corrout = np.zeros(kmax)

    # set up arrays
    y = x.conj().T

    # First half
    for k in range(kmax // 2):

        # calculate acf, need to normalise in case NaNs are present
        corrout[k] = np.nanmean(x[0:k+1] * y[y.size - 1 - k:]) * (k + 1)
    
    # if there is a mid point
    corrout[k + 1] = np.nanmean(x * y) * (x.size)

    # second half
    corrout[k + 2:] = corrout[-k - 3::-1]

    return corrout




## [ AUTO-CORRELATION FUNCTION ] ##
def acf(x, outs = "unique"):
    """
    Calculate normalised Auto-Correlation function using 'fft' method of 
    real-valued data. If NaN values are present, the acf function will use a direct
    summation approach that ignores any NaN values.

    Parameters
    ----------
    x : np.ndarray
        data array, 1D vector
    outs : str, optional
        describes output of acf function, by default "unique" \n
        [unique] - Take positive acf lags, exclude zero-lag peak
        [all] - Take full acf, including positive, negative and zero lags

    Returns
    -------
    np.ndarray
        Auto-Correlation of x data
    """

    # if nan values exist, use "direct" method, will use sum method instead
    if np.any(np.isnan(x)):
        acorr = _nanacf(x)
    else:
        #correlate using FFT approach
        acorr = correlate(x,x,mode = "full", method = "fft")
    
    acorr /= acorr[acorr.size // 2]
    
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
    real-valued data. Does NOT support NaN values

    Parameters
    ----------
    x : np.ndarray
        data array, 1D
    y : np.ndarray
        data array, 1D
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
