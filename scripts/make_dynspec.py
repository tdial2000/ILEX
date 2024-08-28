##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     15/01/2024                 #
# Date (updated):     11/03/2024                 #
##################################################
# Make Dynspecs of stokes IQUV [make_dynspecs.py]#          
#                                                #
# This script makes dynamic spectra of stokes    #
# IQUV with baseline corrections.                #
##################################################

## Imports 
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from math import ceil

## import basic libraries
import argparse, sys
from os import path, mkdir
import shutil
from ilex.htr import make_ds, pulse_fold, baseline_correction
from ilex.data import average



def get_args():
    """
    Info:
        Get arguments passed during script call


    Args:
        args: Arguments for POLCAL.py script

    """

    parser = argparse.ArgumentParser(
        description = "make Dynamic Spectrum of X channel size with full baseline correction."
    )

    ## data arguments
    parser.add_argument("-x", help = "X pol time series", type = str)
    parser.add_argument("-y", help = "Y pol time series", type = str)
    parser.add_argument("--nFFT", help = "Number of frequency channels for final dynspec", 
                        type = int, default = 336)
    parser.add_argument("--bline", help = "Apply baseline correction", action = "store_true")
    parser.add_argument("--QUV", help = "make full stokes dynamic spectra", action = "store_true")


    ## data reduction arguments
    parser.add_argument("--rough_bline", help = "Apply rough baseline by averaging over entire buffer", action = "store_true")
    parser.add_argument("--sigma", help = "S/N threshold for baseline correction", type = float, default = 5.0)
    parser.add_argument("--baseline", help = "Width of rms crops in [ms]", type = float, default = 50.0)
    parser.add_argument("--tN", help = "Time averaging factor, helps with S/N calculation", type = int, default = 50)
    parser.add_argument("--guard", help = "Time between rms crops and burst in [ms]",
                        type = float, default = 1.0)


    ## Pulsar arguments (Polarisation calibration)
    parser.add_argument("--pulsar", help = "Is HTR products of a pulsar", action = "store_true")
    parser.add_argument("--MJD0", help = "Initial Epoch MJD", type = float, default = None)
    parser.add_argument("--MJD1", help = "Observation MJD", type = float, default = None)
    parser.add_argument("--F0", help = "Initial Epoch pulsar frequency", type = float, default = None)
    parser.add_argument("--F1", help = "Spin-down rate", type = float, default = None)
    parser.add_argument("--DM", help = "Dispersion Measure of Pulsar", type = float, default = None)
    parser.add_argument("--cfreq", help = "Central Frequency", type = float, default = 1271.5)
    parser.add_argument("--bw", help = "bandwidth", type = float, default = 336.0)


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




def plot_bline_diagnostic(ds, rbounds, args):
    """
    Generate plot of baseline correction performed

    """

    # create figure and axes
    fig, AX = plt.subplots(2, 1, figsize = (8, 12))
    AX = AX.flatten()
    
    ## calculate time resolution
    dt = 1e-3 * (ds.shape[0]/336)

    ## ms/ or 1000 x dt -> ds time bin converter
    get_units = lambda t : int(ceil(t/dt))

    # get full rbounds and baseline crop as well as a bit of leg room
    crop_start = rbounds[0] - get_units(args.guard + 1.2*args.baseline)
    crop_end = rbounds[1] + get_units(args.guard + 1.2*args.baseline)

    # crop
    ds_crop = average(ds[:,crop_start:crop_end], axis = 1, N = args.tN)

    # get time series
    t_crop = np.mean(ds_crop, axis = 0)

    # get time axis in ms/ or 1000 x dt
    x_crop = np.linspace(0, dt*args.tN * t_crop.size, t_crop.size)


    ## plot
    AX[0].plot(x_crop, t_crop, color = 'k')
    ylim = AX[0].get_ylim()
    AX[0].plot([0.2*args.baseline, 0.2*args.baseline], ylim, 'r--')
    AX[0].plot([1.2*args.baseline, 1.2*args.baseline], ylim, 'r--')
    AX[0].plot([x_crop[-1] - 0.2*args.baseline, x_crop[-1] - 0.2*args.baseline], ylim, 'r--')
    AX[0].plot([x_crop[-1] - 1.2*args.baseline, x_crop[-1] - 1.2*args.baseline], ylim, 'r--')
    AX[0].get_xaxis().set_visible(False)
    AX[0].get_yaxis().set_visible(False)
    AX[0].set_xlim([x_crop[0], x_crop[-1]])
    AX[0].set_ylim(ylim)

    # dynspec plot
    AX[1].imshow(ds_crop, aspect = 'auto', extent = [0, x_crop[-1], 0, 336])
    AX[1].plot([0.2*args.baseline, 0.2*args.baseline], [0,  336], 'r--')
    AX[1].plot([1.2*args.baseline, 1.2*args.baseline], [0, 336], 'r--')
    AX[1].plot([x_crop[-1] - 0.2*args.baseline, x_crop[-1] - 0.2*args.baseline], [0, 336], 'r--')
    AX[1].plot([x_crop[-1] - 1.2*args.baseline, x_crop[-1] - 1.2*args.baseline], [0, 336], 'r--')
    AX[1].set_xlabel("Time [ms]")
    AX[1].set_ylabel("Bandwidth [MHz]")

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0)
    

    # save plot
    plt.savefig(f"{args.ofile}_bline_plot.png")



    







def _proc(args, pol):
    """
    Main processing function
    """

    # initialise parameters
    sphase = None       # starting phase
    rbounds = None      # bounds for baseline correction

    STOKES = "I"
    if args.QUV:
        STOKES = "IQUV"

    print(f"Making the following dynamic spectra: {STOKES}")
    
    # loop over full stokes suite
    for S in STOKES:

        # make dynamic spectra
        ds = make_ds(pol['X'], pol['Y'], S, args.nFFT)

        # remove first channel (zero it)
        ds[0] *= 1e-12

        ## fold if a pulsar has been inputted
        if args.pulsar:
            ds, sphase = pulse_fold(ds, args.DM, args.cfreq, args.bw, args.MJD0, args.MJD1, 
                                      args.F0, args.F1, sphase)

        if args.bline or args.rough_bline:
            ## get baseline corrections
            bs_mean, bs_std, rbounds = baseline_correction(ds, args.sigma, args.guard,
                                            args.baseline, args.tN, args.rough_bline, rbounds)


            ## Apply baseline corrections
            ds[1:] -= bs_mean[1:, None]
            ds[1:] /= bs_std[1:, None]

            # plot baseline diagnostic
            if (S == "I") and (not args.rough_bline):
                plot_bline_diagnostic(ds, rbounds, args)

        ## save data
        print(f"Saving stokes {S} dynamic spectra...")
        np.save(f"{args.ofile}_{S}.npy", ds)



if __name__ == "__main__":
    # main block of code

    ## get args
    args = get_args()


    ## load data
    pol = load_data(args.x ,args.y)


    ## make dynamic spectra
    _proc(args, pol)


    print("Completed!")