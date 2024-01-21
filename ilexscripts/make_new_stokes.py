#!/usr/bin/env python3
##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     05/01/2024                 #
# Date (updated):     05/01/2024                 #
##################################################
# Make new stokes data products                  #
# [make_new_stokes.py]                           #
#                                                #
# This script makes new stokes products whilst   #
# applying a number of processes such as:        #
# DM, baseline correction                        #  
##################################################

## importsx
import argparse
from FRB.FRBhtr import make_stokes
import numpy as np












def get_args():
    """
    Get arguments passed during script call


    ##==== outputs ====##
    args:               Arguments for make_new_stokes.py

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
    parser.add_argument("--f0", help = "reference frequency [MHz]", type = float, default = None)

    ## proc arguments
    parser.add_argument('--DM', help = "Dispersion Measure [pc/cm3]", type = float, default = 0.0)
    parser.add_argument('--t_crop', help = "Crop in time", nargs = 2, type = float)
    parser.add_argument('--f_crop', help = "Crop in frequency", nargs = 2, type = float)

    ## functional arguments
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

    print(args.t_crop)
    print(args.f_crop)

    # run update_XY
    make_stokes(X, Y, fname = args.ofile, t_crop = args.t_crop, 
                f_crop = args.f_crop, BLK_S=1e9, nwork=4, stk="")
    
    return
















if __name__ == "__main__":
    ## main block of code

    ## get arguments
    args = get_args()


    ## run main process
    make_XY(args)

    print("Completed!")

