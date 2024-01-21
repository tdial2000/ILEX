#!/usr/bin/env python3

## imports
import numpy as np
import matplotlib.pyplot as plt


## ilex packages
from ilex.data import *         # all functions to process and manipulate stokes IQUV data
from ilex.plot import *         # suite of plotting functions for IQUV data
from ilex.fitting import *      # fitting functions



#=============================================
# load data
stk = {}
for S in "IQUV":
    stk[S] = np.load(f"/fred/oz002/tdial/FRBdata/FRB_210912/htr/polcal_calib_{S}.npy",
                          mmap_mode = 'r')




#=============================================
# manipulate data and plot

# lets crop and average stokes I
# crop in time
tlim = [0.0, 1.0]
I = pslice(stk['I'], *tlim, axis = 1)

# average in freq
# I = average(I, axis = 0, N = 4)
# average in Time
# I = average(I, axis = 1, N = 100)

# plot dynamic spectra
fig = plot_data(I, "ds")



#==============================================
# calculate PA and plot

tlim = [0.485, 0.495]
flim = [0.05, 1.0]
terrlim = [0.3, 0.4]
cfreq = 1271.5
bw = 336
df = 1.0


# get freqs
freqs = calc_freqs(cfreq, bw, df)


# fit for RM
stk_s = {}
for S in "IQU":
    # average in frequency
    ds = pslice(stk[S], *flim, axis = 0)
    ds = average(ds, axis = 0, N = 4)

    stk_s[S] = np.mean(pslice(ds, *tlim, axis = 1), axis = 1)
    stk_s[f"{S}err"] = np.mean(pslice(ds, *terrlim, axis = 1), axis = 1)

# average freqs
freqs = pslice(freqs, *flim)
freqs = average(freqs, N = 4)

RM, RM_err, f0, pa0 = fit_RMsynth(stk_s['I'], stk_s['Q'], stk_s['U'],
                                    stk_s['Ierr'], stk_s['Qerr'], stk_s['Uerr'], freqs)

fig2 = plot_RM(stk_s['Q'], stk_s['U'], stk_s['Qerr'], stk_s['Uerr'], freqs, RM, pa0, f0)



