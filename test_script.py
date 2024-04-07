from ilex.frb import FRB
from ilex.data import *
from ilex.plot import *
import matplotlib.pyplot as plt
import numpy as np
import sys


frb = FRB(name = "FRB220610", cfreq = 1271.5, bw = 336, dt = 50e-3, df = 4, t_crop = [20.0, 24.5],
            f_crop = [1103.5, 1200], verbose = True, terr_crop = [0.0, 15])
frb.load_data(ds_I = "examples/220610_dsI.npy")

print(frb.par)

frb.fit_tscatt(npulse = 1, priors = {'a1': [0.5, 0.8], 'mu1': [21.0, 22.0], 'sig1': [0.1, 1.0], 'tau': [0.01, 2.0]})

sys.exit()

# lets make a simple scalar weight that multiplies the samples in time
# by -1 so we can see it works
# lets plot the before and after 
frb.plot_data("fI", filename = "docs/source/Tutorials/spec_before_W.png")     # before

frb.par.tW.set(W = -1, method = "None")
frb.plot_data("fI", filename = "docs/source/Tutorials/spec_after_W.png")     # after
# NOTE: the None method is used to specify we want to take the values weights.W as 
# the weights


# define a weighting function, we will make use of a constrained boxcar filter.
# def mask_rms(x, minval, maxval):
#     """
#     Mask data based on rms
#     """

#     W = np.ones(x.size)

#     # set anything before minval to zero
#     W[x < minval] = 0.0

#     # set anything after maxval to zero
#     W[x > maxval] = 0.0

#     return W

# # define the values of minval and maxval to use
# frb.par.tW



