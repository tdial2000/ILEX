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
from ilex.htr import coherent_desperse
import argparse



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

