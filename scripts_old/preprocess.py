# imports
import numpy as np 
import matplotlib.pyplot as plt 
from ilex.frb import FRB 
import argparse
from ilex.utils import load_param_file
from save_data import save_data
from copy import deepcopy


def get_args():

    desc = """ Pre-process data in config file using a number of processes. \n
               1. RFI subtraction: Done by taking equal windows on either side of a reference point (preferably the FRB), adding them together, then subtracting from the data.\n\n
               You can save the data in-place or as a different file. Any output files will be cropped
    
    """

    parser = argparse.ArgumentParser(description = desc)

    # processes
    parser.add_argument('--rfisub', help = "Perform RFI subtraction", action = "store_true")
    parser.add_argument('--rfir', help = "RFI reference point [ms]. If none given will use peak of Stokes I data", type = float, default = None)
    parser.add_argument('--rfig', help = "distance between RFI reference point and RFI windows [ms]", type = float, default = 3.0)
    parser.add_argument('--rfiw', help = "Width of RFI windows [ms]", type = float, default = 5.0)

    # baseline correction
    parser.add_argument("--normf", help = "Normalize frequency channels to get rid of potential baseline errors. Uses --rfi options", action = "store_true")



    # IO
    parser.add_argument('--parfile', help = "ILEX config file (.yaml)", type = str, required = True)
    parser.add_argument('--oparfile', help = "Output ILEX config file with new data, by default will take the value of args.of", type = str, default = None)
    parser.add_argument('--ofile', help = "Output filenames <of>_dsI.npy for stokes I data etc.", type = str, default = None)
    # parser.add_argument('--keep_original_data', help = "Do not perform cropping/downsampling etc. based on ILEX config file, keep original inputs", action = "store_true")


    return parser.parse_args()






def preprocess(args):
    """
    Preprocess data
    """






    ### SAVE DATA
    for 






# get working later on
def RFIsubtraction(args, frb, ds):
    """
    Perform RFI subtraction

    Parameters
    ----------
    args : dict
        Script command args
    frb : FRB
        FRB class
    ds : dict
        Dictionary of stokes dynamic spectra
    """


    # get data
    # get first window
    rfi_lhs = frb.get_data(args.loadedstk, t_crop = [args.rfir - args.rfig - args.rfiw, args.rfir - args.rfig],
                            terr_crop = None, get = True)
    rfi_rhs = frb.get_data(args.loadedstk, t_crop = [args.rfir + args.rfig, args.rfir + args.rfig + args.rfiw],
                            terr_crop = None, get = True)
    
    # subtract RFI
    for s in "IQUV":
        if f"ds{s}" in args.loadedstk
            # average rfi and subtract
            rfi_spec = np.mean(np.concatenate((rfi_lhs[f"ds{s}"], rfi_rhs[f"ds{s}"]), axis = 1), axis = 1)
            ds[s] -= rfi_spec[:, None]

    return









if __name__ == "__main__":

    args = get_args()

