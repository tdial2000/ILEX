##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     11/03/2024                 #
# Date (updated):     11/03/2024                 #
##################################################
# Coherently dedisperse X and Y complex data     #          
#                                                #
##################################################
import numpy as np
import argparse




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



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-x', help = "X polarisation", type = str)
    parser.add_argument('-y', help = "Y polarisation", type = str)
    parser.add_argument('--DM', help = "Dispersion measure", type = float)
    parser.add_argument('--cfreq', help = "central frequency", type = float)
    parser.add_argument('--bw', help = "bandwidth", type = int)
    parser.add_argument('--f0', help = "Reference frequency", type = float)
    parser.add_argument('--quick', help = "apply dispersion using zero-padding", action = "store_true")
    parser.add_argument('-o', help = "Output name", type = str)

    args = parser.parse_args()

    return args




if __name__ == "__main__":

    # get function args
    args = get_args()

    # load data
    xpol = np.load(args.x, mmap_mode = 'r')
    ypol = np.load(args.y, mmap_mode = 'r')


    # apply dispersion on X
    t_des = coherent_desperse(xpol, args.cfreq, args.bw, args.f0, args.DM, args.quick)

    # save
    np.save(f"{args.o}_X.npy", t_des)

    # apply dispersion on Y
    t_des = coherent_desperse(ypol, args.cfreq, args.bw, args.f0, args.DM, args.quick)

    # save 
    np.save(f"{args.o}_Y.npy", t_des)

