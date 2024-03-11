##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     20/06/2023                 #
# Date (updated):     11/03/2024                 #
##################################################
# Quickly plot dynamic spectra                   #          
#                                                #
##################################################
#import
import argparse, os
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    #arguments
    parser = argparse.ArgumentParser(description = "plot dynamic spectrum")

    # ext params
    parser.add_argument("--tN", help = "time scrunching", type = int, default = 1)

    ## FILENAME ##
    parser.add_argument("specfile",help = "filename of dynamic spectrum")

    # parse arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    dynspec = np.load(args.specfile)

    # scrunch
    if args.tN > 1:
        t_new = (dynspec.shape[1] // args.tN) * args.tN
        dynspec = np.mean(dynspec[:,:t_new].reshape(dynspec.shape[0], t_new // args.tN, args.tN), axis = 2)



    #plot data 
    plt.figure("Dynamic Spectrum",figsize = (10,10))
    plt.imshow(dynspec,aspect = 'auto', extent=[0,1,0,1])

    plt.show()
