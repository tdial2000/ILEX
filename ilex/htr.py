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








def make_ds(xpol, ypol, S = "I", nFFT = 336):
    """
    Make dynamic spectra from complex X, Y polarisation time series arrays

    Parameters
    ----------
    xpol : np.ndarray or array-like
        X time series polarisation data
    ypol : np.ndarray or array-like
        Y time series polarisation data
    S : str, optional
        type of Stokes dynamic spectra to make, options are ['I', 'Q', 'U', 'V'], by default "I"
    nFFT : int, optional
        number of frequency channels in final dynamic spectrum, by default 336

    Returns
    -------
    ds: np.ndarray
        dynamic spectrum
    """    

    prog_str = f"[Stokes] = {S} with [nFFT] = {nFFT}"

    Stk_Func = {"I":stk_I, "Q":stk_Q, "U":stk_U, "V":stk_V}

    # pre-processing for iterative 
    BLOCK_SIZE = 200e6 # block size in B
    BIT_SIZE = 8       # Bit size in B

    # First need to chop data so that an integer number of FFT windows can
    # be performed. Afterwards, this data will be split into coarse BLOCKS
    # with specific memory constraints. 

    # define parameters
    nsamps  = xpol.size                  # original number of samples in loaded dataset
    fnsamps = (nsamps // nFFT) * nFFT    # number of samples after chopping 
    nwind   = fnsamps // nFFT            # number of fft windows along time series

    # memeory block paramters
    nwinb = int(BLOCK_SIZE // (nFFT * BIT_SIZE))    # num windows in BLOCK
    nsinb = int(nwinb * nFFT)                       # num samples in BLOCK
    nblock= int(nwind // nwinb)                     # num BLOCKS

    # create empty array
    ds = np.zeros((nFFT, nwind), dtype = np.float32)

    b_arr = np.empty((0,2), dtype = int)
    for i in range(nblock):
        b_arr = np.append(b_arr, [[i*nwinb,(i+1)*nwinb]], axis = 0)
    # append extra block at end
    if nblock*nsinb < nsamps:
        b_arr = np.append(b_arr, [[(i+1)*nwinb,nwind]], axis = 0)

    # loop over blocks
    for i, b in enumerate(b_arr): # b is bounds of block in nFFT windows
        sb = b * nFFT
        wind_w = b[1] - b[0]
        ds[:,b[0]:b[1]] = Stk_Func[S](fft(xpol[sb[0]:sb[1]].copy().reshape(wind_w, nFFT), axis = 1),
                                 fft(ypol[sb[0]:sb[1]].copy().reshape(wind_w, nFFT), axis = 1)).T
        
        # print progress
        print(f"[MAKING DYNSPEC]:    [Progress] = {(i+1)/(nblock+1)*100:3.3f}%:    " + prog_str,
              end = '         \r')

    print("[MAKING DYNSPEC]:    [Progress] = 100.00%:    " + prog_str + "        \n")
    print(f"Made Dynamic spectra with shape [{ds.shape[0]}, {ds.shape[1]}]")

    

    return ds







def pulse_fold(ds, DM, cfreq, bw, MJD0, MJD1, F0, F1, sphase = None):
    """
    Takes Pulsar dynamic spectrum and folds it, removes periods
    at far left and right sides to avoid band artifacts produced during
    de-dispersion.

    Parameters
    ----------
    ds : np.ndarray or array-like 
        dynamic spectrum
    MJD0 : float
        Initial Epoch MJD
    MJD1 : float
        Observation MJD
    F0 : float 
        initial Epoch Frequency period
    F1 : float 
        Spin-down rate
    sphase : float 
        Starting phase of folding, if not given will be estimated (best done using "I" ds)

    Returns
    -------
    ds_f : np.ndarray 
    Raw folded Dynamic spectra
    sphase : float
    Starting phase of folding from original dynspec

    """
    print("Pulse Folding Dynspec...")
    # normalise needed to help find fold bounds
    ds_mean = np.mean(ds, axis = 1)[:, None]
    ds_std = np.std(ds, axis = 1)[:, None]

    ## Calculate Period T in [s]
    T = 1/(F0 - F1 * (MJD1 - MJD0)*86400)
    print(f"with period T = {T}")
    dt = 1e-6 * (ds.shape[0]/336) # get time resolution of dynspec

    ## Fold dynamic spectra
    fold_w = int(T / dt)          # fold width in samples (assumed dt = 1 us)
    fold_n_init = int(ds.shape[1]/fold_w)     # initial number of folds

    # get dispersion sweep, calculate number of "broken" pulse periods
    # due to dipsersion.
    top_band = cfreq + bw/2
    bot_band = cfreq - bw/2
    DM_sweep = 4.14938e3 * DM * (1/bot_band**2 - 1/top_band**2) # DM sweep in seconds
    P_sweep = int(DM_sweep/T) + 1
    print(f"DM sweep: {DM_sweep} [ms]")
    print(f"Culling {P_sweep} Periods to the left due to DM sweep")



    # find index of peak in second period, then get starting phase
    if sphase is None:
        search_crop = ds[:,P_sweep*fold_w:(P_sweep + 1)*fold_w]
        search_crop = (search_crop - ds_mean)/ds_std
        pulse2i = np.mean(search_crop, axis = 0).argmax()
        pulse2i += P_sweep * fold_w
        sphase = pulse2i - int(fold_w/2)
    else:
        # put in sample units
        sphase = int(sphase*ds.shape[1])

    # calculate number of folds
    fold_n = int((ds.shape[1]-(sphase+1))/fold_w)     # number of folds
    print(f"Folding {fold_n}/{fold_n_init} periods")


    # reshape to average folds together
    # ignore side periods due to dedispersing
    ds_r = ds[:,sphase:sphase + fold_w * (fold_n)].copy()
    ds_f = np.mean(ds_r.reshape(ds_r.shape[0], (fold_n), fold_w), axis = 1)

    
    return ds_f, sphase / ds.shape[1]














def baseline_correction(ds, sigma: float = 5.0, guard: float = 1.0, 
                        baseline: float = 50.0, tN: int = 50, rbounds = None):
    """
    Get baseline corrections to the Dynamic spectra data

    Parameters
    ----------
    ds : np.ndarray or array-like 
        Dynamic spectra
    sigma : float 
        S/N threshold for bounds
    guard : float 
        Time in [ms] between rough bounds and rms crop region
    baseline : float 
        Width of buffer in [ms] to estimate baseline correction
    tN : int 
        Time Averaging factor for Dynamic spectrum, helps with S/N calculation.
    rbounds : list 
        Bounds of FRB burst, if Unspecified, the code will do a rough S/N calculation 
        to determine a bursts bounds

    Returns
    -------
    bs_mean : np.ndarray or array-like 
        Baseline mean
    bs_std : np.ndarray or array-like 
        Baseline std
    rbounds : np.ndarray or array-like 
        Bounds of FRB burst in Phase units

    """      

    print("Applying baseline correction...")
    # rough normalisation needed to find burst
    ds_n = (ds - np.mean(ds, axis = 1)[:, None])/(np.std(ds, axis = 1)[:, None])

    # static parameters
    rmsg = 0.5   # rms guard in phase difference from peak of burst

    ## calculate time resolution
    dt = 1e-3 * (ds.shape[0]/336) 

    ## ms -> ds time bin converters
    get_units_avg = lambda t : int(ceil(t/(dt * tN)))
    get_units = lambda t : int(ceil(t/dt))

    ## find burst
    if rbounds is None:
        ## Rough normalize 
        ds_r = average(ds_n, axis = 1, N = tN)
        rmean = np.mean(ds_r, axis = 1)
        rstd = np.std(ds_r, axis = 1)

        ds_rn = ds_r - rmean[:, None]
        ds_rn /= rstd[:, None]

        
        ## find burst bounds
        print("Looking for bounds of burst...")
        # get peak, crop rms and do rough S/N calculation
        t_rn = np.mean(ds_rn, axis = 0)
        peak = np.argmax(t_rn)
        rms_w = get_units_avg(baseline)
        rms_crop = np.roll(t_rn, int(rmsg * ds_rn.shape[1]))[peak-rms_w:peak+rms_w]
        rms = np.mean(rms_crop**2)**0.5

        # calculate S/N
        t_sn = t_rn / rms
        rbounds = np.argwhere(t_sn >= sigma)[[0,-1]]/t_sn.size
        rbounds = np.asarray((rbounds*ds.shape[1]), dtype = int)[:,0]

    ## calculate baseline corrections
    
    guard_w = get_units(guard)
    rms_w = get_units(baseline)
    lhs_crop = ds[:,rbounds[0]-guard_w-rms_w:rbounds[0]-guard_w]
    rhs_crop = ds[:,rbounds[1]+guard_w:rbounds[1]+guard_w+rms_w]
    bl_crop = np.concatenate((lhs_crop, rhs_crop), axis = 1)


    bs_mean = np.mean(bl_crop, axis = 1)
    bs_std = np.std(bl_crop, axis = 1)


    return bs_mean, bs_std, rbounds











def coherent_desperse(t, cfreq, bw, f0, DM, fast = False, 
                      DM_iter = 50):
    """
    Apply Coherent dedespersion on Complex Polarisation time series data

    Parameters
    ----------
    t : np.mmap or np.ndarray
        Complex Polarisation time series data
    cfreq : float
        Central Frequency of observing band [MHz]
    bw : float
        Bandwidth of observation [MHz]
    f0 : float
        Reference Frequency
    DM : float
        Dispersion Measure [pc/cm^3]
    fast : bool, optional
        Apply FFT and IFFT quickly by zero-padding data to optimal length. , by default False \n
        Note: This shouldn't affect results too much assuming full CELEBI HTR data, however, the longer 
        the padding relative to the original size of dataset, the worse the data, so use wisely.
    DM_iter : int, optional
        Number of iterations to split Dispersion into, by default 50

    Returns
    -------
    t_d : np.ndarray
        De-dispersed Complex Polarisation times series data
    """

    prog_str = f"[bw] = {bw} MHz with [cfreq] = {cfreq} MHz:    [DM] = {DM} pc/cm3 at a ref freq [f0] = {f0} MHz"

    # constants
    kDM = 4.14938e3         # DM constant

    next_len = t.size
    if fast:
        next_len = next_fast_len(next_len)

    # ifft
    t_d = fft(t, next_len)

    # apply dispersion
    uband = cfreq + bw/2                            # upper band
    bw = bw*(next_len/t.size)                       # updated bandwidth due to zero padding in time domain

    BLOCK_size = int(next_len / DM_iter)            # number of samples per DM iteration
    BLOCK_end = next_len - BLOCK_size * DM_iter     # in case number of samples don't divide into DM_iter evenly, usually
                                                    # a handful of samples are left at the end, this is BLOCK_end
    BLOCK_bw = float(bw*BLOCK_size/next_len)        # amount of bandwidth being covered per DM iteration

    # iterate over chuncks to save memory
    for i in range(DM_iter):
        freqs = (np.linspace(uband - i*BLOCK_bw, uband - (i+1)*BLOCK_bw, BLOCK_size, endpoint = False) 
                +bw/next_len/2)

        # disperse part of t series
        t_d[i*BLOCK_size:(i+1)*BLOCK_size] *= phasor_DM(freqs, DM, f0)

        print(f"[DISPERSING]:    [Progress] = {(i+1)/(DM_iter+1)*100:3.3f}%:    " + prog_str, end = "\r")

    # apply last chunck if needed
    if BLOCK_end > 0:
        freqs = np.linspace(uband - (i+1)*BLOCK_bw,uband - bw, BLOCK_end, endpoint = False) +bw/next_len/2

        # disperse
        t_d[(i+1)*BLOCK_size:] *= phasor_DM(freqs, DM, f0)

    print("[DISPERSING]:    [Progress] = 100.00%:    " + prog_str + "\n")

    # inverse fourier tranform back to time domain
    t_d = ifft(t_d, next_len)[:t.size]

    return t_d




# def coherent_scatter():
#     """
    
#     """
#     pass





# def coherent_rotate():
#     """

#     """
#     pass
