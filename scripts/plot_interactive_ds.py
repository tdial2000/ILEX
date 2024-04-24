##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     20/04/2024                 #
# Date (updated):     20/04/2024                 #
##################################################
# make interactive dynspec                       #          
#                                                #
##################################################
## imports
from ilex.script_core.plot_interactive_ds import plot_interactive_ds
import matplotlib.pyplot as plt 
import argparse

# empty class
class _empty:
    pass


def get_args():
    """ get arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument("--parfile", help = "parameter file containning info about FRB", type = str)
    parser.add_argument("-S", help = "Stokes parameter", type = str, default = "I")

    return parser.parse_args()




if __name__ == "__main__":
    # main block of code

    args = get_args()

    # run interactive plot
    fig, return_ = plot_interactive_ds(parfile = args.parfile, S = args.S)

    # plot
    plt.show()