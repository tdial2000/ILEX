#!/usr/bin/env python3

## Imports 
import numpy as np


## import basic libraries
import argparse, sys
from os import path, mkdir


def get_args():
    """
    Get arguments passed during script call


    ##==== outputs ====##
    args:               Arguments for POLCAL.py script

    """

    parser = argparse.ArgumentParser(
        description = "Pulse fold stokes dynamic spectra"
    )

    ## data arguments
    parser.add_argument("-i", help = "Stokes I dynspec", type = str)
    parser.add_argument("-q", help = "Stokes Q dynspec", type = str, default = None)
    parser.add_argument("-u", help = "Stokes U dynspec", type = str, default = None)
    parser.add_argument("-v", help = "Stokes V dynsepc", type = str, default = None)


    ## reduction arguments
    parser.add_argument("--stripends", help = "Strip periods at the end in case significant de-dipserison distored them", action = "store_true")


    ## observation arguments
    parser.add_argument("--T", help = "Pulse Period", type = float)


    ## output arguments
    parser.add_argument("--ofile", help = "Common prefix for folded output products", type = str)

    args = parser.parse_args()

    return args





def load_data(args):
    """
    Load in Stokes I, Q, U & V data along with 
    estimates on L/I and V/I coefficients.

    ##== inputs ==##
    args:           Arguments of POLCAL.py script


    ##== outputs ==##
    stk:            dict of stokes dynspecs [IQUV]
    freqs:          array of frequencies
    l_model:        L/I model for calibrator
    v_model:        V/I model for calibrator
    
    """

    ## load in stokes data
    stk = {}

    stk['I'] = np.load(args.i, mmap_mode = "r")

    if args.q is not None:
        stk['Q'] = np.load(args.q, mmap_mode = "r")
    
    if args.u is not None:
        stk['U'] = np.load(args.u, mmap_mode = "r")

    if args.v is not None:
        stk['V'] = np.load(args.v, mmap_mode = "r")


    return stk




def fold(args, stk):
    """
    Fold dynamic spectra
    """

    ## create new dict for folded and scrunched stokes dynspecs
    stk_f = {}

    # iterate over "IQUV" items
    for S in stk.keys():
        print(f"          Folding/scrunching Stokes {S}...")
        
        ##==================##
        ## fold and scrunch ##
        ##==================##

        ## fold data (presumably a pulsar) by some period 
        fold_w = int(args.T * 1e6)          # fold width in samples (assumed dt = 1 us)
        fold_n = int(stk[S].shape[1]/fold_w)     # number of folds

        # find index of peak in second period
        if S == "I":
            pulse2i = np.mean(stk[S][:,fold_w:2*fold_w], axis = 0).argmax()
            pulse2i += fold_w
            sphasei = pulse2i - int(args.T * 0.5e6)

        # reshape to average folds together (Also remove top band TODO)
        # ignore side periods due to dedispersing
        ds_r = stk[S][:,sphasei:sphasei + fold_w * (fold_n-2)].copy()
        stk_f[S] = np.mean(ds_r.reshape(ds_r.shape[0], (fold_n-2), fold_w), axis = 1)

    
    return stk_f




if __name__ == "__main__":
    # main code to run

    # get arguments
    args = get_args()


    # load data
    stk = load_data(args)


    # fold data
    stk_fold = fold(args, stk)


    # save folded data
    for S in stk_fold.keys():
        print(f"Saving Folded stokes {S} dynspec")
        np.save(f"{args.ofile}_{S}.npy", stk_fold[S])


    print("Completed!")