##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Date created: 26/11/2024 
## Last updated: 26/11/2024
##
## 
## 
## Functions for estimating the position, widths
## and bounds of a signal (i.e. an FRB) 
##===============================================##
##===============================================##
# imports
import numpy as np
from scipy.signal import correlate
from .data import average, pslice
import matplotlib.pyplot as plt


def find_optimal_sigma_width(tI, sigma: int = 5, rms_guard: float = 0.033, 
                                rms_width: float = 0.0667, rms_offset: float = 0.33):
    """
    This function searches the stokes I dynamic spectrum for the most likely
    location of the frb. It's important to note that this function will look through
    the entire dataset regardless of crop parameters. It will first scrunch, so if memory
    is an issue first set 'tN'.

    Parameters
    ----------
    sigma: int 
        S/N threshold
    rms_guard: float 
        gap between estiamted pulse region and 
        off-pulse region for rms and baseband estimation, in [phase units]
    rms_width: float 
        width of off-pulse region on either side of pulse region in [phase units]
    rms_offset: float 
        rough offset from peak on initial S/N threshold in [phase units]
    **kwargs: 
        FRB parameters + FRB meta-parameters
    
    Returns
    -------
    peak: int
        index of peak value in burst
    lw: int
        lower bound width from peak
    hw: int 
        upper bound width from peak

    """

    peak = np.argmax(tI)

    rms_guard = int(rms_guard * tI.size)
    rms_width = int(rms_width * tI.size)
    rms_offset = int(rms_offset * tI.size)

    # estimate rough rms and hence rough bounds of burst
    if (peak - rms_offset - rms_width < 0) or (peak + rms_offset + rms_width > tI.size - 1):
        print("[rms_offset] and/or [rms_width] out of bounds of [tI]!! Aborting")
        return (None)*3

    rms_lhs = tI[peak - rms_offset - rms_width : peak - rms_offset]
    rms_rhs = tI[peak + rms_offset : peak + rms_offset + rms_width]
    rough_rms = np.mean(np.concatenate((rms_lhs, rms_rhs))**2)**0.5

    signal = np.where(tI / rough_rms > sigma)[0]

    rough_lowerbound, rough_upperbound = np.min(signal), np.max(signal)


    # recalculate optimal rms
    if (peak - rms_guard - rms_width < 0) or (peak + rms_guard + rms_width > tI.size - 1):
        print("[rms_guard] and/or [rms_width] out of bounds of [tI]!! Aborting")
        return (None)*3

    rms_lhs = tI[rough_lowerbound - rms_guard - rms_width : rough_lowerbound - rms_guard]
    rms_rhs = tI[rough_upperbound + rms_guard : rough_upperbound + rms_guard + rms_width]
    optimal_rms = np.mean(np.concatenate((rms_lhs, rms_rhs))**2)**0.5

    signal = np.where(tI / optimal_rms > sigma)[0]

    optimal_lowerbound, optimal_upperbound = np.min(signal), np.max(signal)


    # calculate lhs and rhs widths w.r.t peak
    lw = peak - optimal_lowerbound
    rw = optimal_upperbound - peak
    return peak, lw, rw





def find_optimal_fluence_width(tI, yfrac = 0.95, mode = "median"):
    """
    Find optimal width/bounds of frb by finding the 95% cutoff on either
    side of the effective centroid.

    Parameters
    ----------
    tI : np.ndarray or array-like
        Stokes I time series profile
    yfrac : float
        fraction of total fluence on either side of FRB effective centroid to take
        as FRB bounds
    mode : str
        type of algorithm to use when finding optimal fluence width \n
        [median] -> find burst width by estimating centroid of burst and fluence threshold on either side \n
        [min] -> find minimum burst width that captures the desired fluence threshold (moving window algorithm)

    Returns
    -------
    centroid : int
        index of effective centroid of tI
    lw : int
        effective yfrac width on the LHS of centroid
    rw : int
        effective yfrac width on the RHS of centroid
    
    """


    # Check data first
    if (yfrac < 0.0) or (yfrac > 1.0):
        raise ValueError("yfrac must be between [0.0, 1.0]")
    
    if mode not in ["median", "min"]:
        raise ValueError(f"mode = {mode} invalid, must be either 'median' or 'min'")
    
    print(f"Finding optimal [{mode}] burst width and centroid")

    # perform burst width search
    if mode == "median":
        centroid, lw, rw = _find_median_fluence_width(tI, yfrac)
    elif mode == "min":
        centroid, lw, rw = _find_min_fluence_width(tI, yfrac)
    

    return centroid, lw, rw






def _find_median_fluence_width(tI, yfrac = 0.95):
    """
    Find median fluence width
    """

    # calculate effective centroid of burst
    fluence = np.sum(tI)
    centered_cumsum = np.cumsum(tI) - fluence/2
    centroid = np.argmin(np.abs(centered_cumsum))
    

    # find yfrac points of LHS and RHS of centroid

    # LHS
    lhs_ind = np.argmin(np.abs(centered_cumsum + yfrac * fluence/2))
    lw = centroid - lhs_ind

    # RHS
    rhs_ind = np.argmin(np.abs(centered_cumsum - yfrac * fluence/2))
    rw = rhs_ind - centroid

    return centroid, lw, rw




def _find_min_fluence_width(tI, yfrac = 0.95):    
    
    # fulence and starting window
    fluence = np.sum(tI)
    N = 1

    def find_N_length(tI, Nstart, Nstep):

        N = Nstart

        while True:
            corr = correlate(tI, 1/fluence * np.ones(N), mode = "valid")
            p = np.where(corr >= yfrac)[0]
            if p.size > 0:
                if N == 1:
                    print("The window offset found was 1, there may be something wrong with the data or input data is too small?")
                    return N, p
                if Nstep == 1:
                    if p.size > 1:
                        print("There appears to be two centroids to a minimum width.")
                    return N, p
                if Nstep > 1:
                    N, p = find_N_length(tI, N - Nstep, Nstep // 10)
                
                break
            N += Nstep

        return N, p

    # get minimum width of burst
    N, p = find_N_length(tI, N, 100)
    print(f"Found fluence threshold at N = {N}")

    # find centroid of burst
    tI_burst = tI[p[0]:p[0] + N]
    burst_fluence = np.sum(tI_burst)
    cumsum = np.cumsum(tI_burst)
    centroid = np.argmin(np.abs(cumsum - burst_fluence/2))

    # output centroid in original time series frame and LHS RHS width w.r.t centroid
    lw = centroid 
    rw = N - centroid
    centroid += p[0]

    return centroid, lw, rw






def find_optimal_sigma_dt(tI, sigma: float = 15.0, rms_offset: float = 0.33, rms_width: float = 0.0667):
    """

    Parameters
    ----------
    tI : np.ndarray or array-like
        time series
    sigma : int, optional
        minimum peak Signal-to-noise, by default 15

    Returns
    -------
    tN : int
        averaging factor needed to reach desired peak Signal-to-noise threshold
    """

    tN = 1

    # loop over tNs
    try:
        while True:
            # downsample and calculate peak S/N
            tI_avg = average(tI, N = tN)

            peak = np.argmax(tI_avg)
            peak_val = tI_avg[peak]
            peak = float(peak)/float(tI_avg.size)
            tI_rms = pslice(tI_avg, peak - rms_offset - rms_width, peak - rms_offset)
            rms = np.mean(tI_rms**2)**0.5

            peak_sigma = peak_val / rms

            if peak_sigma >= sigma:
                print(f"Maximum time resolution found at [{tN} * dt] where dt is the intrinsic resolution.")
                print(f"Peak S/N: {peak_sigma}")
                break
        
            tN += 1
    except:
        print(f"Something went wrong, possibly a peak S/N of [{sigma}] could not be reached. ")
        print(f"Last checked averaging factor: [{tN}]")


    return tN       
