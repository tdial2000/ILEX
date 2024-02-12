##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     15/01/2024                 #
# Date (updated):     16/01/2024                 #
##################################################
# Make Dynspecs of stokes IQUV                   #          
#                                                #
# This script makes dynamic spectra of stokes    #
# IQUV with the following optional               #
##################################################

## Imports 
import numpy as np
from copy import deepcopy
from scipy.fft import fft
from math import ceil

## import basic libraries
import argparse, sys
from os import path, mkdir
import shutil

# add ILEX path
sys.path.insert(1, '../')

# import from htr ilex package
from ilex.htr import *








def get_args():
    """
    Info:
        Get arguments passed during script call


    Args:
        args: Arguments for POLCAL.py script

    """

    parser = argparse.ArgumentParser(
        description = "Make Dynamic spectra with additional functionality"
    )

    ## data arguments for making dynamic spectra
    parser.add_argument("-x", help = "X pol time series", type = str)
    parser.add_argument("-y", help = "Y pol time series", type = str)
    parser.add_argument("--nFFT", help = "Number of frequency channels for final dynspec", 
                        type = int, default = 336)
    parser.add_argument("--nwork", help = "Number of workers to perform stft", default = 1)
    parser.add_argument("--BLOCKSIZE", help = "Size of blocks in Bytes to split data into", 
                        type = float, default = 200e6)
    parser.add_argument("--BITSIZE", help = "BIT size in Bytes of data", type = float, default = 8)
    parser.add_argument("--fast", help = "Perform quicker 1D FFTs using by zero-padding data", action = "store_true")
    parser.add_argument("--stks", help = "Additional stokes dynspecs to make besides I, i.e. --stk 'QUV'", type = str,
                        default = "")
    parser.add_argument("--notnegateQ", help = "Negate Q due to ASKAP PAF", action = "store_false")




    ## data arguments for De-dispersion
    parser.add_argument("--DM", help = "Dispersion Measure [pc/cm3]", type = float, default = None)
    parser.add_argument("--cfreq", help = "Central Frequency [MHz]", type = float, default = 1271.5)
    parser.add_argument("--f0", help = "Reference Frequency [MHz]", type = float, default = 1103.5)
    parser.add_argument("--bw", help = "Bandwidth in [MHz]", type = float, default = 336.0)
    parser.add_argument("--DM_iter", help = "Number of iterations to split process into", type = int,
                        default = 50)
    parser.add_argument("--saveXY", help = "Save De-dispersed X and Y POL products", action = "store_true")
    
    


    ## data arguments for baseline correction
    parser.add_argument("--bline", help = "Apply baseline correction", action = "store_true")
    parser.add_argument("--sigma", help = "S/N threshold for baseline correction", type = float, default = 5.0)
    parser.add_argument("--baseline", help = "Width of rms crops in [ms]", type = float, default = 50.0)
    parser.add_argument("--tN", help = "Time averaging factor, helps with S/N calculation", type = int, default = 50)
    parser.add_argument("--guard", help = "Time between rms crops and burst in [ms]",
                        type = float, default = 10.0)
    parser.add_argument("--rmsmp", help = """Phase difference between maximum point in time and mid
                       point of rms crop used to estimate rough initial S/N""", type = float, default = 0.5)
    parser.add_argument("--dt", help = "Time resolution in [ms] of dynamic spectra", type = float, default = 0.001)




    ## Pulsar arguments (Polarisation calibration)
    parser.add_argument("--pulsar", help = "Is HTR products of a pulsar", action = "store_true")
    parser.add_argument("--MJD0", help = "Initial Epoch MJD", type = float, default = None)
    parser.add_argument("--MJD1", help = "Observation MJD", type = float, default = None)
    parser.add_argument("--F0", help = "Initial Epoch pulsar frequency", type = float, default = None)
    parser.add_argument("--F1", help = "Spin-down rate", type = float, default = None)


    ## output arguments
    parser.add_argument("--ofile", help = "Name of new dynamic spectra", type = str)

    args = parser.parse_args()

    return args





def load_data(xfile, yfile):
    """
    Info:
        Load in Stokes I, Q, U & V data along with 
        estimates on L/I and V/I coefficients.

    Args
        args: Arguments of POLCAL.py script


    Return
        stk (dict): dict of stokes dynspecs [IQUV]
    
    """

    ## load in stokes data
    pol = {}

    pol['X'] = np.load(xfile, mmap_mode = 'r')
    pol['Y'] = np.load(yfile, mmap_mode = 'r')


    return pol

















if __name__ == "__main__":
    # main block of code


    # get arguments
    args = get_args()


    # load data
    pol = load_data(args.x, args.y)


    # Apply Dispersion
    if args.DM is not None:
        print("Applying De-Disperison:")
        print(f"DM: {args.DM}   [pc/cm3]")
        print(f"cfreq: {args.cfreq}    [MHz]")
        print(f"f0: {args.f0}   [MHz]")
        print(f"bw: {args.bw}    [MHz]")

        # apply dispersion on x data
        print("Dispersing [X] Polarisation")
        pol['X'] = coherent_desperse(pol['X'], cfreq = args.cfreq, bw = args.bw,
                    f0 = args.f0, DM = args.DM, fast = args.fast, DM_iter = args.DM_iter)
        
        # apply dispersion on y data
        print("Dispersing [Y] Polarisation")
        pol['Y'] = coherent_desperse(pol['Y'], cfreq = args.cfreq, bw = args.bw,
                    f0 = args.f0, DM = args.DM, fast = args.fast, DM_iter = args.DM_iter)

        if args.saveXY:
            print(f"Saving [X,Y] Pol time series...")
            np.save(f"{args.ofile}_X.npy", pol['X'])
            np.save(f"{args.ofile}_Y.npy", pol['Y'])

    
    # Make Spectra
    # init params
    rbounds = None
    sphase = None

    stks = "I" + args.stks
    for S in stks:
        print(f"Stokes {S}")

        ## make stokes data
        print("[1] - Making spectra")
        ds = make_stokes(pol['X'], pol['Y'], stokes = S, nFFT = args.nFFT, nworkers = args.nwork,
                    BLOCK_SIZE = args.BLOCKSIZE, BIT_SIZE = args.BITSIZE, negateQ = args.notnegateQ)

        # fold if a pulsar has been inputted
        if args.pulsar:
            print("[2] - Pulse folding")
            ds, sphase = pulse_fold(ds, MJD0 = args.MJD0, MJD1 = args.MJD1, F0 = args.F0,
                            F1 = args.F1, sphase = sphase)

        # apply baseline correction
        
        if args.bline:
            print("[3] - Baseline correction")
            ds, rbounds = baseline_correction(ds, sigma = args.sigma, guard = args.guard, 
                            baseline = args.baseline, rmsmp = args.rmsmp, tN = args.tN, dt = args.dt,
                            rbounds = rbounds)

        # save dynamic spectra to file
        print(f"Saving stokes {S} dynamic spectra...")
        np.save(f"{args.ofile}_{S}.npy", ds)