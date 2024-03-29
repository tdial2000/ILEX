#!/usr/bin/env python

from ilex.frb import FRB
from ilex.data import *
from ilex.plot import *
import matplotlib.pyplot as plt
import numpy as np
import subprocess
# from FRB.FRBstats import scatt_pulse_profile
# from FRB.FRBstats import get_bestfit
import sys, os

#testing loading
frb = "190102"
DM = 364.5
dynI_file = f"/fred/oz002/askap/craft/craco/processing/output/{frb}/htr/{frb}_I_dynspec_{DM}.npy"
dynQ_file = f"/fred/oz002/askap/craft/craco/processing/output/{frb}/htr/{frb}_Q_dynspec_{DM}.npy"
dynU_file = f"/fred/oz002/askap/craft/craco/processing/output/{frb}/htr/{frb}_U_dynspec_{DM}.npy"
dynV_file = f"/fred/oz002/askap/craft/craco/processing/output/{frb}/htr/{frb}_V_dynspec_{DM}.npy"

# dynI_file = f"/fred/oz002/tdial/FRBdata/FRB_190711/htr/190711_calib_10khz_I.npy"
# dynQ_file = f"/fred/oz002/tdial/FRBdata/FRB_190711/htr/190711_calib_10khz_Q.npy"
# dynU_file = f"/fred/oz002/tdial/FRBdata/FRB_190711/htr/190711_calib_10khz_U.npy"
# dynV_file = f"/fred/oz002/tdial/FRBdata/FRB_190711/htr/190711_calib_10khz_V.npy"



# xpol_file = "/fred/oz002/askap/craft/craco/processing/output/230708/htr/230708_X_t_411.51.npy"
# ypol_file = "/fred/oz002/askap/craft/craco/processing/output/230708/htr/230708_Y_t_411.51.npy"

# dynI_file2 = "/fred/oz002/tdial/FRBdata/FRB_230708/beamformed/dynspec_0.10.npy"

dynI_file = "/fred/oz002/tdial/FRBdata/FRB_230708/beamformed/230708_calib_I.npy"
dynQ_file = "/fred/oz002/tdial/FRBdata/FRB_230708/beamformed/230708_calib_Q.npy"
dynU_file = "/fred/oz002/tdial/FRBdata/FRB_230708/beamformed/230708_calib_U.npy"
dynV_file = "/fred/oz002/tdial/FRBdata/FRB_230708/beamformed/230708_calib_V.npy"

dynI_file = "/fred/oz002/tdial/HTR_paper_data/230708/polcal_I.npy"
dynQ_file = "/fred/oz002/tdial/HTR_paper_data/230708/polcal_Q.npy"
dynU_file = "/fred/oz002/tdial/HTR_paper_data/230708/polcal_U.npy"
dynV_file = "/fred/oz002/tdial/HTR_paper_data/230708/polcal_V.npy"

dynI_file = "/fred/oz002/askap/craft/craco/processing/output/220610/htr/220610_I_dynspec_1457.624.npy"
dynQ_file = "/fred/oz002/askap/craft/craco/processing/output/220610/htr/220610_Q_dynspec_1457.624.npy"
dynU_file = "/fred/oz002/askap/craft/craco/processing/output/220610/htr/220610_U_dynspec_1457.624.npy"
dynV_file = "/fred/oz002/askap/craft/craco/processing/output/220610/htr/220610_V_dynspec_1457.624.npy"

dynI_file = "/fred/oz002/tdial/ilex/examples/220610_dsI.npy"
dynQ_file = "/fred/oz002/tdial/ilex/examples/220610_dsQ.npy"
dynU_file = "/fred/oz002/tdial/ilex/examples/220610_dsU.npy"
dynV_file = "/fred/oz002/tdial/ilex/examples/220610_dsV.npy"

# dynI_file = "/fred/oz002/tdial/HTR_paper_data/211212/211212_I.npy"
# dynQ_file = "/fred/oz002/tdial/HTR_paper_data/211212/211212_Q.npy"
# dynU_file = "/fred/oz002/tdial/HTR_paper_data/211212/211212_U.npy"
# dynV_file = "/fred/oz002/tdial/HTR_paper_data/211212/211212_V.npy"


frb = FRB(verbose = True)
frb.set(cfreq = 1271.5, bw = 336, dt = 50e-3, df = 4, t_crop = [20.9, 23.8], f_crop = [1103.5, 1200], plot_err_type = 'regions')
frb.load_data(ds_I = dynI_file, ds_Q = dynQ_file, 
              ds_U = dynU_file, ds_V = dynV_file)

# frb.plot_data("dsI", filename = "/fred/oz002/tdial/ilex/docs/source/Tutorials/220610_dsI_crop.png")
# get time series profile 
priors = {'a1': [0.5, 0.8], 'mu1': [21.0, 22.0], 'sig1': [0.1, 1.0], 'tau': [0.01, 2.0]}
frb.set(t_crop = [20.9, 23.8], f_crop = [1103.5, 1200])     # set crop params
p = frb.fit_RM(method = "RMsynth", terr_crop = [0, 15], t_crop = [21.4, 21.6])
frb.plot_PA(terr_crop = [0, 15], plot_L = True, plot = True, filename = "/fred/oz002/tdial/ilex/docs/source/Tutorials/220610_PA.png")
# p.plot(filename = "/fred/oz002/tdial/ilex/docs/source/Tutorials/220610_RM.png")
# p = frb.fit_tscatt(method = "bayesian", plot = True, priors = priors)
# p.stats()
# tI = frb.get_data(data_list = ["tI"], get = True)['tI']     # get data

# # make x axis array
# x = np.linspace(*frb.par.t_lim, tI.size)

# # plot 
# plt.figure(figsize = (12,8))
# plt.plot(x, tI)
# plt.xlabel("Time [ms]")
# plt.ylabel("Flux Density (arb.)")
# plt.savefig("/fred/oz002/tdial/ilex/docs/source/Tutorials/220610_tI_crop.png")
# plt.show()
# frb.load_data(ds_I = dynI_file)
# frb.save_data(["dsI", "dsQ", "dsU", "dsV"], name = "/fred/oz002/tdial/ilex/examples/220610")

# frb.fit_RM(terr_crop = [0, 35], method = "RMsynth")

# frb.load_data(ds_I = dynI_file, ds_Q = dynQ_file, 
#               ds_U = dynU_file, ds_V = dynV_file)

# frb.plot_PA(terr_crop = [0, 35])

# frb.fit_RM(terr_crop = [0, 35], method = "RMsynth")

# frb.plot_PA(terr_crop = [0, 35], plot_L = True)


# frb.save_data(data_list = ["fI", "fQ", "fU", "fV", "dsQ", "tV"], name = "/fred/oz002/tdial/test_ILEX_MAKE/vela")

# plt.figure()
# freq = np.load("/fred/oz002/tdial/test_ILEX_MAKE/vela_freq.npy")
# for S in "IQUV":
#     dat = np.load(f"/fred/oz002/tdial/test_ILEX_MAKE/vela_f{S}.npy")
#     plt.plot(freq,dat, label = S)

# plt.legend()
# plt.show()

# frb.plot_err_type = "regions"
# frb.plot_data("dsI")
# frb.fit_RM(t_crop = [2320, 2335], terr_crop = [2200, 2300], plot = True, method = "RMsynth", fN = 4, fit_params = {'showPlots': True})
# frb.plot_stokes(t_crop = [40, 52], terr_crop = [0, 35], stk_type = "t", tN = 10, plot_L = False)
# frb.plot_poincare(t_crop = [40, 52], terr_crop = [0, 35], fN = 4, stk_type = "t", tN = 10, plot_model = True, 
# normalise = True, sigma = 5)

# frb.plot_PA(t_crop = [40, 52], terr_crop = [0, 35], tN = 10, plot_L = True, RM = 41, verbose = True)
# frb.plot_PA_multi(tcrops = [[40,45],[45,52]], terr_crop = [0, 35], tN = 10, plot_L = True, RM = None, verbose = True)



sys.exit()
