##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     17/03/2024                 #
# Date (updated):     17/03/2024                 #
##################################################
# make multi tile plot                           #          
#                                                #
##################################################

## imports
from ilex.frb import FRB
from ilex.data import *
from ilex.utils import load_param_file, dict_get
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def get_args():
    """
    Get args

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--parfile", help = "Parameter file containning info about FRB", type = str)
    parser.add_argument("--filename", help = "Save to file", type = str, default = None)
    return parser.parse_args()


def plot_stokes(args):
    """
    Plot stokes time and dynamic spectra
    
    """

    fig, AX = plt.subplot_mosaic("S;I;Q;U;V", figsize = (10, 12),
                gridspec_kw = {"height_ratios":[1,1,1,1,1]}, sharex = True)
    
    # load in parfile
    frb = FRB()
    frb.load_data(yaml_file = args.parfile)

    # get data
    data = frb.get_data(["tI", "tQ", "tU", "tV", "dsI", "dsQ", "dsU", "dsV"], get = True)

    # get pars
    pars = load_param_file(args.parfile)

    # plot stokes time series
    frb.plot_stokes(ax = AX['S'], stk_type = "t", plot_L = pars['plots']['plot_L'],
        Ldebias = pars['plots']['Ldebias'], debias_threshold = pars['plots']['debias_threshold'])
    AX['S'].set(ylabel = "Flux Density (arb.)")

    # plot dynspec
    extent = [*frb.this_par.t_lim, *frb.this_par.f_lim]
    xw = data['time'][-1] - data['time'][0]
    yw = abs(data['freq'][-1] - data['freq'][0])
    for S in "IQUV":
        AX[S].imshow(data[f"ds{S}"], aspect = 'auto', 
            extent = extent)
        AX[S].set(ylabel = "Freq [MHz]")
        # add textbox
        AX[S].text(data['time'][0] + 0.97*xw, np.min(data['freq']) + 0.95*yw,
                S, fontsize = 16, verticalalignment = 'top', bbox = {'boxstyle':'round', 
                'facecolor':'w', 'alpha':0.7})


    AX['V'].set(xlabel = "Time offset [ms]")

    # final figure adjustments
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0)

    return fig




if __name__ == "__main__":
    # main block of code to run

    args = get_args()

    # plot stokes
    fig = plot_stokes(args)

    # save figure
    if args.filename is not None:
        plt.savefig(args.filename)
    
    plt.show()
