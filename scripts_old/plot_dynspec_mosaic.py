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
from ilex.script_core.plot_dynspec_mosaic import plot_dynspec_mosaic
import matplotlib.pyplot as plt
import argparse


def get_args():
    """
    Get args

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--parfile", help = "Parameter file containning info about FRB", type = str)
    parser.add_argument("-t", help = "Integration times", nargs='+', default = [1, 3, 10, 30, 100, 300, 1000], type = int)
    parser.add_argument("--nsamp", help = "halfwidth of crop to take around maximum point, in samples", type = int, default = 100)
    parser.add_argument("--tN", help = "Averaging factor in time, help find maximum and align spectrum", type = int, default = 10)
    parser.add_argument("--defaraday_ds", help = "De-faraday rotate dynamic spectra", action = "store_true")
    parser.add_argument("--filename", help = "Save to file", type = str, default = None)

    return parser.parse_args()




if __name__ == "__main__":
    # main block of code

    args = get_args()

    # plot large mosaic
    fig = plot_dynspec_mosaic(parfile = args.parfile, t = args.t, nsamp = args.nsamp, tN = args.tN,
                    defaraday_ds = args.defaraday_ds, filename = args.filename)

    plt.show()

    
