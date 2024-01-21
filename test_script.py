#!/usr/bin/env python

from ilex.frb import FRB
from ilex.data import average
import matplotlib.pyplot as plt
import numpy as np
# from FRB.FRBstats import scatt_pulse_profile
# from FRB.FRBstats import get_bestfit
import sys

#testing loading
frb = "190102"
DM = 361.53
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

# dynI_file = "/fred/oz002/tdial/FRBdata/FRB_210912/htr/polcal_calib_I.npy"
# dynQ_file = "/fred/oz002/tdial/FRBdata/FRB_210912/htr/polcal_calib_Q.npy"
# dynU_file = "/fred/oz002/tdial/FRBdata/FRB_210912/htr/polcal_calib_U.npy"
# dynV_file = "/fred/oz002/tdial/FRBdata/FRB_210912/htr/polcal_calib_V.npy"

dynI_file = f"/fred/oz002/tdial/testing/make_XY/{frb}_I.npy"
dynQ_file = f"/fred/oz002/tdial/testing/make_XY/{frb}_Q.npy"
dynU_file = f"/fred/oz002/tdial/testing/make_XY/{frb}_U.npy"
dynV_file = f"/fred/oz002/tdial/testing/make_XY/{frb}_V.npy"

# dynI_file = "/fred/oz002/tdial/FRBdata/FRB_210912/htr/210912_calib_no_offset_I.npy"
# dynQ_file = "/fred/oz002/tdial/FRBdata/FRB_210912/htr/210912_calib_no_offset_Q.npy"
# dynU_file = "/fred/oz002/tdial/FRBdata/FRB_210912/htr/210912_calib_no_offset_U.npy"
# dynV_file = "/fred/oz002/tdial/FRBdata/FRB_210912/htr/210912_calib_no_offset_V.npy"

# class cls:



#     def func(self, a = None):

#         if a is None:
#             a = {}

#         print(a)
#         b = a.copy()
#         b["a"] = 12


# cl = cls()

# cl.func()

# cl.func()

# def func(*args):

#     out_ = list(args)
    
#     for i,item in enumerate(out_):
#         out_[i] = {}

#     if len(out_) == 1:
#         out_ = out_[0]
    
#     return out_

# a = 12
# b = 12
# c = 45
# v: dict = None
# b: dict = None
# v,b = func(v,b)

# print(v)
# print(b)




# sys.exit()



frb = FRB(name = "210912",bw = 336, cfreq = 1271.5, tN = 5, verbose = True)



frb.load_data(ds_I = dynI_file, ds_Q = dynQ_file,
              ds_U = dynU_file, ds_V = dynV_file)
# frb.plot_data("dsI", t_crop = [1943., 1945.])
# frb.plot_stokes(stk_type = 'f', t_crop = [1943.436, 1943.750], terr_crop = [0.67, 0.684])
# frb.plot_PA(t_crop = [1943.436, 1943.750], terr_crop = [0.67, 0.684], fN = 8, f0 = 1271.5)
frb.plot_PA_multi(method = "RMquad", tcrops = [[1943.436, 1943.750], [1943.846, 1944.179]],
                     terr_crop = [0.67, 0.684], fN = 8, Ldebias_threshold=1.0, plot_L = True)


# print(frb)
# sys.exit()
# frb.fit_RM(method = "RMquad", terr_crop = [0.4, 0.45], f0 = 1271.5, t_crop = [1919.424, 1922.318], fN = 8)
# frb.find_frb()
# frb.plot_stokes(stk_type = 't', t_crop = [1941, 1945], terr_crop = [0.67, 0.684])
# frb.plot_PA(method = "RMquad",t_crop = [1943.25, 1944.25], terr_crop = [0.67, 0.684], f0 = 1271.5, plot_L = False, Ldebias_threshold=1.5, fN = 4, RM = -109, pa0 = -0.918)
# frb.plot_data("fU", tN = 1, t_crop = [1665.813, 1666.572], f_crop = [1104, 1420], fN = 4, terr_crop = [0.5, 0.53])

# frb.fit_RM(method = "RMquad", t_crop = [1943.25, 1944.25], terr_crop = [0.67, 0.684], fN = 1)

# frb.get_data(data_list = ["dsI", "tI", "fI", "dsQ", "dsU", "dsV", "tV"], terr_crop = [0.4, 0.5])
# frb.test_log()





# frb.plot_stokes(stk_ratios = False, t_crop = [1665.747, 1668.264], f_crop = [1104, 1420], tN = 5, L = False, fN = 5, stk_type="t")
# frb.plot_data(data = "tI", t_crop = [1665.747, 1668.264], f_crop = [1104, 1420], tN = 10)
# frb.fit_RM(method = "RMsynth", plot = True, tN = 10, t_crop = [0.549, 0.551], f_crop = [1110, 1420], fN = 5, terr_crop = [0.5, 0.53])

# dsIerr = frb.get_data(["dsI"], t_crop = [0.2, 0.5], f_crop = [1110, 1420], get = True)["dsI"]
# dsI = frb.get_data(["dsI"], t_crop = [0.549, 0.551], get = True, f_crop = [1110, 1420])["dsI"]

# # calculate channel error
# fig = plt.figure(figsize = (10,10))

# for N in [1, 5, 10, 100]:
#     dsI_avg = average(dsIerr, axis = 1, N = N)
#     # tIerr = np.std(dsI_avg, axis = 1)
#     tIerr = np.std(dsI_avg, axis = 1)/(dsI_avg.shape[1])**0.5
#     plt.plot(np.arange(tIerr.size), tIerr/np.mean(dsI, axis = 1), label = str(N))

# plt.plot(np.arange(tIerr.size), np.mean(dsI, axis = 1)/ np.mean(dsI, axis = 1), 'k--')

# plt.legend()
# plt.show()



# frb.find_frb()
# frb.plot_PA(terr_crop = [0.4, 0.5], t_crop = [0.549, 0.551], Ldebias_threshold=4, plot_L = False, RM = 5.0359, f_crop = [1104, 1420], f0 = 1363.0986, pa0 = -1.5025031)
# # print(frb)
# print(frb)
# frb.plot_PA(terr_crop = [0.4, 0.5], t_crop = [0.68207, 0.68787], Ldebias_threshold=3, plot_L = True)

# sys.exit()

sys.exit()
# frb.metapar.t_crop = frb.find_frb()

# frb.get_data(data_list = "all", terr_crop = [0.4, 0.5], norm = "None")

# f_err = frb.get_data(data_list = ["dsI", "tI"], get = True, t_crop = [0.4, 0.5], norm = "None", terr_crop = [0.2, 0.5])["dsI"]
# print(frb)


# print(np.mean(f_err**2, axis = 1)**0.5)
# print(np.std(f_err, axis = 1))

# plt.figure()
# # plt.imshow(f_err, aspect = 'auto')
# plt.plot(f_err[-1]**2)
# # print(np.std(f_err[-1]))

# plt.show()
# sys.exit()

# find frb
# frb.plot_data(data = "fI", t_crop = [0.4, 0.5], norm = "None")


# frb.plot_data(data = "fI", terr_crop = [0.4, 0.5], norm = "None")
# frb.plot_ds_int(stk = "Q", t_crop = [0.68, 0.69])








# sys.exit()


# frb.plot_data(data = "tI", f_crop = [0.0, 1.0])

# sys.exit()

# print(frb.par)

# frb.plot_dyn_int()

# frb.plot_data(data = "tI")

# get dynamic spectra of stokes I
# dynI = frb.get_data(data = "dynI")
# tI = frb.get_data(data = "tI")
# fI = frb.get_data(data = "fI")
# frb.norm = True
# frb.t_crop = [0.68173,0.6829]
# print(frb.t_crop)
# tI2 = frb.get_data(data = "tI")
# tI2 = frb.get_data(data = "fI")

# sys.exit()
# plt.figure()
# plt.imshow(dynI, aspect = 'auto')

# plt.figure()
# plt.plot(tI)

# plt.figure()
# plt.plot(fI)


# plt.show()
# print(frb.t_crop)

# frb.plot_data("tI", t_crop = [0.68173, 0.6829], norm = "None")

# sys.exit()

priors = {"a1": [0.75,0.95], "a2": [0.0,0.2], "tau": [0.0,1.0],"sig1": [0.05,0.07],"mu1": [2124.538,2125.838],"mu2":[2125.187,2125.887],
          "mu3":[2125.375,2126.075],"sig2":[0.02,0.1],"sig3":[0.04,0.08],"sigma":[0,0.1],"a3":[0.7,1.0]}

static_priors = { "mu1":2125.149, "mu2":2125.5194, "mu3":2125.7972, "sig1":0.0653, "sig2":0.0502, "sig3": 0.0598, "sigma": 0.002}
static_priors = {}

p = frb.fit_tscatt(fit_priors = priors, static_priors = static_priors,
                                     npulse = 3, plot = True,fit_params={"npool":32,"label":"out75"}, t_crop = [0.68173, 0.6829], norm = "None", tN = 10)

# modl_y = scatt_pulse_profile(x, {key:p[key].val for key in p.keys()})
# modl_y /= np.max(modl_y)

sys.exit()
# print(modl_y.shape)

# plt.figure()
# # plt.plot(x, modl_y)
# # plt.show()
# # print(frb.t_crop)
# tI = frb.get_data(data = "tI")

# # print(tI.shape)

# tI_weighted = frb.get_data(data = "tI") * modl_y

# plt.plot(tI, label = "norm")
# plt.plot(tI_weighted, label = "weighted")

# plt.show()
frb.load_data(ds_I = dynI_file2)
frb.metapar.tN = 1
frb.metapar.t_crop = [0.682,0.6826]
# modl_y = modl_y[84:modl_y.size - 93]
print(modl_y.size)

plt.figure()
plt.plot(modl_y)

p, x, y, fig = frb.get_scintband(plot = True, fit_params = {"npool": 8, "label": "scintnew12"})

print(p['a'].val, p['a'].plus)
plt.show()
sys.exit()
# loop over subbands
N = 4
dN = 1.0 / N
output = "scint_sub2"
subband_scint = np.zeros(N)
for i in range(N):
    print(f"subband {i}")
    frb.metapar['f_crop'] = [dN*(i), dN*(i+1)]
    p, x, y, fig = frb.get_scintband(plot = True, fit_params = {"npol": 8, "label": f"subband{i}", "outdir":output}, tW = modl_y)
    subband_scint[i] = p["w"].val






# frb230708.par.df = 0.1

# print(frb230708.norm)

# frb230708.load_data(dynI_file = dynI_file2, ypol_file = ypol_file,
#                      xpol_file = xpol_file)

# print(frb230708.par.df)


# # print(frb230708)

# frb230708.plot_dynI()

# # # print out get_tscatt()
# priors = {"a1": [0.7,0.9], "a2": [0.0,0.2], "tau": [0, 1.0],"sig1": [0.05,0.07],"mu1": [2124.538,2125.838],"mu2":[2125.187,2125.887],
#           "mu3":[2125.375,2126.075],"sig2":[0.04,0.08],"sig3":[0.04,0.08],"sigma":[0,0.1],"a3":[0.6,0.9], "a4":[0.7, 0.9]}
# priors_ls = {"a1":0.9,"a2":0.1,"a3":0.8,"mu1":2125.238,"mu2":2125.507,"mu3":2125.875,"sig1":0.07,"sig2":0.07,"sig3":0.07,"tau":0.3}
# priors_ls2 = {"a1":0.9,"a2":0.7,"mu1":2125.238,"mu3":2125.875,"sig1":0.07,"sig2":0.07,"sig3":0.07,"tau":0.5}
# static_priors = { "mu1":2125.149, "mu2":2125.5194, "mu3":2125.7972, "sig1":0.0653, "sig2":0.0502, "sig3": 0.0598, "sigma": 0.002}




# p, x, y, fig = frb230708.get_scintband(method = "bays", plot = True, fit_params = {"npool": 4, "label": "scint13"})
# p, x, y, fig = frb230708.get_tscatt(method = "ls", plot = True, fit_params = {"maxfev":20000}, fit_priors = priors_ls, npulse = 3, static_priors = static_priors)
# p, x, y, fig = frb230708.get_tscatt(fit_priors = priors, static_priors = static_priors, method = "bays", npulse = 3, plot = True,fit_params={"npool":8,"label":"out12"})

# static_wrap_fit_func("lorentz", {"a": 0.0, "w": 1.0}, {"w": 1.5})

