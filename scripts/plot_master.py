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
from ilex.script_core.plot_master import plot_master
import argparse
import matplotlib.pyplot as plt

def get_args():
    """
    Get arguments
    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--parfile", help = "Parameter file containing info about FRB", type = str)
    parser.add_argument("--plot_panels", default = "[S;D]",
                         help = "types of plots, [S;D] means plot Stokes on top panel and Dynamic spectrum on bottom")
    parser.add_argument("--model", help = "Model time series plot", action = "store_true")
    parser.add_argument("--modelpar", help = "Use param file to model data", type = str, default = None)
    parser.add_argument("--modelpulses", help = "Plot each individual pulse", action = "store_true")
    parser.add_argument("--filename", help = "Save to file", type = str, default = None)

    return parser.parse_args()













if __name__ == "__main__":
    # main block of code

    args = get_args()
    
    fig = plot_master(parfile = args.parfile, plot_panels = args.plot_panels, 
                      model = args.model, modelpar = args.modelpar, 
                      modelpulses = args.modelpulses, filename = args.filename)

    plt.show()