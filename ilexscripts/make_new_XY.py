#!/usr/bin/env python3
##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     03/01/2024                 #
# Date (updated):     03/01/2024                 #
##################################################
# Make new XY data products [make_new_XY.py]     #
#                                                #
# This script makes new X,Y products whilst      #
# applying a number of processes such as:        #
# DM, RM, leakage etc.                           #  
##################################################

## imports
import argparse
from FRB.FRBhtr import update_XY
import numpy as np












def get_args():
    """
    Get arguments passed during script call


    ##==== outputs ====##
    args:               Arguments for POLCAL.py script

    """

    parser = argparse.ArgumentParser(
        description = "Fit for pol cal solutions"
    )

    ## input arguments
    parser.add_argument('-x', help = "Filename of X pol", type = str)
    parser.add_argument('-y', help = "Filename of Y pol", type = str)

    ## observation arguments
    parser.add_argument("--cfreq", help = "Central frequency [MHz]", type = float)
    parser.add_argument("--bw", help = "Bandwidth [MHz]", type = float)

    ## calibrator arguments
    parser.add_argument("--pa0", help = "PA at reference frequency [rad]", type = float, default = 0.0)
    parser.add_argument("--f0", help = "reference frequency [MHz]", type = float, default = None)

    ## proc arguments
    parser.add_argument('--DM', help = "Dispersion Measure [pc/cm3]", type = float, default = 0.0)
    parser.add_argument('--RM', help = "Rotation Measure [rad/m2]", type = float, default = 0.0)
    parser.add_argument('--tau', help = "leakage Time delay [us]", type = float, default = 0.0)
    parser.add_argument('--phi', help = "leakage Phase delay [rad]", type = float, default = 0.0)
    parser.add_argument('--psi', help = "Rotational offset in (X,Y)", type = float, default = 0.0)

    ## functional arguments
    parser.add_argument('--iter', help = "Number of blocks to split into", default = 50, type = int)
    parser.add_argument('--fast', help = "Apply fast FFT", action = "store_true")

    ## output arguments
    parser.add_argument('--ofile', help = "Common prefix for new X,Y filenames", type = str)


    args = parser.parse_args()

    return args
















def make_XY(args):
    """
    Wrapper function for making new X, Y products

    ##==== inputs ====##
    args:         arguments

    ##==== outputs ====##
    None:

    """

    # load data
    print("Loading data")
    X = np.load(args.x, mmap_mode = 'r')
    Y = np.load(args.y, mmap_mode = 'r')

    # run update_XY
    update_XY(X = X, Y = Y, filename = args.ofile,
              cfreq = args.cfreq, bw = args.bw, f0 = args.f0,
              DM = args.DM,
              tau = args.tau, phi = args.phi, psi = args.psi,
              fast = args.fast, _iter = args.iter)
    
    return
















if __name__ == "__main__":
    ## main block of code

    ## get arguments
    args = get_args()


    ## run main process
    make_XY(args)

    print("Completed!")

