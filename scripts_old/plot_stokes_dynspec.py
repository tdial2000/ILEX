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
from ilex.script_core.plot_stokes_dynspec import plot_stokes_dynspec
import matplotlib.pyplot as plt
import argparse


def get_args():
    """
    Get args

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--parfile", help = "Parameter file containning info about FRB", type = str)
    parser.add_argument("--filename", help = "Save to file", type = str, default = None)
    return parser.parse_args()




if __name__ == "__main__":
    # main block of code to run

    args = get_args()

    # plot stokes
    fig = plot_stokes_dynspec(parfile = args.parfile, filename = args.filename)
    
    plt.show()
