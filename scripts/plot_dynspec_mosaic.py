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
    parser.add_argument("-t", help = "Integration times", nargs='+', default = [1, 3, 10, 30, 100, 300, 1000], type = int)
    parser.add_argument("--nsamp", help = "halfwidth of crop to take around maximum point, in samples", type = int, default = 100)
    parser.add_argument("--defaraday_ds", help = "De-faraday rotate dynamic spectra", action = "store_true")
    parser.add_argument("--filename", help = "Save to file", type = str, default = None)

    return parser.parse_args()




# plot mosaic
def plot_mosaic(args):
    """
    Plot Mosaic
    
    """
    print(args.t)

    num = len(args.t)
    pmax = max(args.t)

    # create figure
    axes_handles = [[f"{S}{t}" for t in args.t + ["f"]] for S in "tIQUV"]
    x_plot_w = 2*7/num
    fig, AX = plt.subplot_mosaic(axes_handles, figsize = (18,12),
            gridspec_kw = {"height_ratios": [1,2,2,2,2], "width_ratios": [x_plot_w]*num+[1]})



    # create frb instance
    frb = FRB()
    frb.load_data(yaml_file = args.parfile)

    # get params from file
    pars = load_param_file(args.parfile)



    # get bounds of data, find max point of burst and take window around it to plot
    data = frb.get_data(["tI"], get = True, t_crop = [0.0, 1.0], tN = 100)
    Imax = float(np.argmax(data['tI']))/data['tI'].size                         # position of max in burst
    Imax_ms = Imax * (frb.par.t_lim[1] - frb.par.t_lim[0]) + frb.par.t_lim[0]   
    phasew = pmax*args.nsamp/frb.ds['I'].shape[1]                               # width of immediate burst in phase units
    t_crop_full = [Imax - 1.2*phasew, Imax + 1.2*phasew]                        # crop to use in frb instance


    # enumerate through each given time resolution
    for i, t in enumerate(args.t):
        # calculate time crop
        twidth = t*1e-3*args.nsamp
        tcrop = [Imax_ms - twidth, Imax_ms + twidth]
        print(tcrop)

        for S in "IQUV":

            # get dynamic spectra and spectra, here we have the option of derotating it or not
            if not args.defaraday_ds:
                data = frb.get_data([f"ds{S}", f"f{S}"], t_crop = tcrop,
                            RM = 0.0, get = True, tN = t)
            else:
                ds_data = frb.get_data([f"ds{S}", f"f{S}"], t_crop = tcrop,
                            get = True, tN = t)



            # calculate time as an offset from the start of the crop
            t_offset = [val - frb.prev_par.t_lim[0] for val in frb.prev_par.t_lim]



            # plot dynspec
            AX[f"{S}{t}"].imshow(data[f"ds{S}"], aspect = 'auto', 
                            extent = [*t_offset, *frb.prev_par.f_lim])

            # set labels
            if S == "V":
                AX[f"{S}{t}"].set(xlabel = "Time offset [ms]")
            else:
                AX[f"{S}{t}"].get_xaxis().set_visible(False)



            # We only want to make a spectrum of the highest resolution spectrum, i.e. i == 0
            if not i:
                AX[f"{S}{t}"].set(ylabel = "Freq [MHz]")
                AX[f"{S}f"].plot(data[f"f{S}"], data['freq'], color = 'k')
                ylim = frb.prev_par.f_lim
                AX[f"{S}f"].plot([0.0, 0.0], ylim, '--k')
                AX[f"{S}f"].set_ylim(ylim)
            else:
                AX[f"{S}{t}"].get_yaxis().set_visible(False)



            # add text at right most ds labelling the stokes parameter
            if i == len(args.t) - 1:
                xw = t_offset[-1] - t_offset[0]
                yw = abs(data['freq'][-1] - data['freq'][0])
                AX[f"{S}{t}"].text(t_offset[0] + 0.92*xw, np.min(data['freq']) + 0.95*yw,
                    S, fontsize = 16, verticalalignment = 'top', bbox = {'boxstyle':'round', 
                    'facecolor':'w', 'alpha':0.7})



            
        # plot stokes time series profiles
        frb.plot_stokes(ax = AX[f"t{t}"], stk_type = "t", plot_L = pars['plots']['plot_L'], t_crop = tcrop, tN = t,
                Ldebais = pars['plots']['Ldebias'], debias_threshold = pars['plots']['debias_threshold'])
        AX[f"t{t}"].set_title(f"{t} $\\mu$s")
        AX[f"t{t}"].set_xlim(frb.prev_par.t_lim)
        

        # logic for labeling mosaic
        if not i:
            AX[f"t{t}"].set(ylabel = "Flux Density (arb.)")
        else:
            AX[f"t{t}"].get_yaxis().set_visible(False)
        AX[f"t{t}"].get_xaxis().set_visible(False)

        if i < num - 1:
            AX[f"t{t}"].get_legend().remove()
        else:
            AX[f"t{t}"].get_legend().set(bbox_to_anchor=(1.03, 0.95))
        
    for S in "IQUV":
        AX[f"{S}f"].get_yaxis().set_visible(False)
        AX[f"{S}f"].get_xaxis().set_visible(False)
    AX[f"tf"].set_axis_off()
        

    # final figure adjustments
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)  

    return fig







if __name__ == "__main__":
    # main block of code

    args = get_args()

    # plot large mosaic
    fig = plot_mosaic(args)

    # save fig
    if args.filename:
        plt.savefig(args.filename)


    plt.show()

    
