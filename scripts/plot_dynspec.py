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
from ilex.data import average


def get_args():
    #arguments
    parser = argparse.ArgumentParser(description = "plot dynamic spectrum")

    # ext params
    parser.add_argument("--tN", help = "time scrunching", type = int, default = 1)
    parser.add_argument("--fN", help = "freq scrunching", type = int, default = 1)

    ## FILENAME ##
    parser.add_argument("specfile",help = "filename of dynamic spectrum")

    # parse arguments
    args = parser.parse_args()

    return args



def plot_zaps(ax, zapbool):

    zaps = np.ones(zapbool.size, dtype = float) * np.nan
    zaps[zapbool] = 0.55

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xwidth = xlim[1] - xlim[0]
    ax.imshow(zaps.reshape(zaps.size, 1), aspect = 'auto', cmap = "OrRd",
                vmax = 1, vmin = 0, extent = [xlim[0], xlim[0] + 0.02 * xwidth, *ylim])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



if __name__ == "__main__":

    args = get_args()

    dynspec = np.load(args.specfile)

    # scrunch
    if (args.tN > 1) or (args.fN > 1):
        t_new = (dynspec.shape[1] // args.tN) * args.tN
        dynspec = average(dynspec, axis = 1, N = args.tN)
        dynspec = average(dynspec, axis = 0, N = args.fN, nan = True)



    #plot data 
    fig, ax = plt.subplots(2, 1, figsize = (12,10), gridspec_kw = {'height_ratios':[1, 3]}, sharex = True)
    ax = ax.flatten()
    ax[0].plot(np.nanmean(dynspec, axis = 0), 'k')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xlim([0, dynspec.shape[1]])

    zapbool = np.isnan(dynspec[:, 0])
    dynspec[zapbool] = 0.0

    ax[1].imshow(dynspec,aspect = 'auto')
    plot_zaps(ax[1], zapbool)
    ax[1].set_xlabel("Time Bins", fontsize = 16)
    ax[1].set_ylabel("Freq Bins", fontsize = 16)

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)

    plt.show()
