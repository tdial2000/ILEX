##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     17/03/2024                 #
# Date (updated):     17/03/2024                 #
##################################################
# Plot multiple PA components                    #          
#                                                #
##################################################

## imports
from ilex.script_core.plot_PA_multi import plot_PA_multi
import argparse
import matplotlib.pyplot as plt



def get_args():
    """
    Get args
    
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--parfile", help = "parameter file containning info about FRB", type = str)
    parser.add_argument("--RMplots", help = "Show grid of RM fits of each component", action = "store_true")
    parser.add_argument("--RMburst", help = "Plot varaiblility of RM across burst", action = "store_true")
    parser.add_argument("--showbounds", help = "Show bounds of each component", action = "store_true")
    parser.add_argument("--filename", help = "save to file", type = str, default = None)

    # t_crops (Manual)
    parser.add_argument("--start", help = "Start time in [ms] of list of t_crops", default = None, type = float)
    parser.add_argument("--width", help = "Width in [ms] of each t_crop segment", default = None, type = float)
    parser.add_argument("--ncomp", help = "Number of components (number of tcrops)", default = None, type = int)

    return parser.parse_args()





if __name__ == "__main__":
    # main block of code

    args = get_args()

    # plot 
    fig = plot_PA_multi(parfile = args.parfile, RMplots = args.RMplots, RMburst = args.RMburst, 
                        start = args.start, width = args.width, ncomp = args.ncomp,
                        showbounds = args.showbounds, filename = args.filename)

    plt.show()

