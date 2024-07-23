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
from ..frb import FRB
from ..data import *
from ..utils import load_param_file, dict_get, fix_ds_freq_lims
import yaml
import numpy as np
import matplotlib.pyplot as plt


class _empty:
    pass



def plot_stokes_dynspec(parfile, filename = None):


    args = _empty()
    args.parfile = parfile
    args.filename = filename

    fig = _plot_stokes(args)

    return fig






def _plot_stokes(args):
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
    frb.plot_stokes(ax = AX['S'], stk_type = "t",
        Ldebias = pars['plots']['Ldebias'], sigma = pars['plots']['sigma'], stk_ratio = pars['plots']['stk_ratio'],
        stk2plot = pars['plots']['stk2plot'])
    AX['S'].set(ylabel = "Flux Density (arb.)")

    # plot dynspec
    ds_freq_lims = fix_ds_freq_lims(frb.this_par.f_lim, frb.this_par.df)
    extent = [*frb.this_par.t_lim, *ds_freq_lims]
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

    if args.filename is not None:
        plt.savefig(args.filename)

    return fig