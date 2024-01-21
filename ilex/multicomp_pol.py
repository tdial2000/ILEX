##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 21/01/2024 
##
## Multi-component polarisation functions
## PA, RM
## 
##
##===============================================##
##===============================================##
## imports 
from .data import *         # data functions
from .fitting import *      # fitting functions
from .plot import *         # plotting functions
from .logging import log    # logging
from ._frb_proc import _proc_par, _proc_data
import numpy as np
import matplotlib.pyplot as plt
from .utils import plotnum2grid



# NOTE:
# in the case of using the ilex interface and t_lim[0] is something other than 0.0,
# then this functions x axis in time won't be accurate

def multicomp_pol(stk, freqs, method = "RMquad", Ldebias_threshold = 2.0, dt = 0.001, plot_L = False,
                par: dict = None, tcrops = None, fcrops = None, filename: str = None,
                 **kwargs):
    """
    Info:
        Multicomponent Polarisation, does full RM fitting and PA plotting/spectra plotting for each
        component specified by tcrops and fcrops

    Args:
        stk (dict): Dictionary of memory maps for full STOKES IQUV HTR dynamic spectra
                    [I] - Stokes I
                    [Q] - Stokes Q
                    [U] - Stokes U
                    [V] - Stokes V
        freqs (ndarray): Full Frequency array [MHz]
        method (str): Method for RM fitting
                      [RMquad] - Use Quadratic function and Scipy.curve_fit
                      [RMsynth] - Use RM synthesis from [RMtools] from CIRCADA [https://github.com/CIRADA-Tools/RM-Tools]
        Ldebias_threshold (float): Threshold for Debiased linear fraction in PA masking
        dt (float): Time resolution in [ms] of stk data
        tcrops (list): List of Time phase limits for each component
        fcrops (list): List of Freq phase limits for each component
        par (dict): List of parameters for data processing, see _proc_data() in _frb_proc() for a list of params
        plot_L (bool): Plot Linear Fraction (L) instead of stokes Q and U
        filename (str): Save plots to files with a common prefix
        **kwargs: Keyword arguments for fitting processes (see method for RM fitting)

    Returns:
        RMs (list): List of fitted RMs for each components
        PA (ndarray): Position angle for full data
        PA_err (ndarray): Position angle error ""
    
    """
    log(f"Plotting PA for {len(tcrops)} components")

    # check if tcrops and fcrops are same length
    if tcrops is None and fcrops is None:
        log("Must specify time and/or freq crops for each component", stype = "err")
        return (None,) * 3
    elif tcrops is not None and fcrops is not None:
        if len(tcrops) != len(fcrops):
            log("number of time and freq crops must be equal", stype = "err")    
            return (None,) * 3

    # initialise par
    par = _proc_par(par)

    # initialise crops
    if tcrops is None:
        tcrops = [par['t_crop']] * len(fcrops)
        
    if fcrops is None:
        fcrops = [par['f_crop']] * len(tcrops)

    # check if error time crop is given
    if par['terr_crop'] is None:
        log("Must specify Time phase crop for estimating off-pulse errors", stype = "err")
        return (None,) * 3
    

    # initialise useful parameters
    ncomps = len(tcrops)                    # number of components
    tMAX = dt * stk['I'].shape[1]           # amount of time across htr data


    # initialise containers
    RM = [0.0] * ncomps
    RM_err = [0.0] * ncomps
    f0 = [par['f0']] * ncomps
    pa0 = [0.0] * ncomps
    PA = []
    PA_err = []


    # RM parameters
    if method == "RMquad":
        log("Fitting RM to Quadratic function")
        data_list = ['fQ', 'fU']
    elif method == "RMsynth":
        log("Fitting RM using RM synthesis")
        data_list = ["fI", "fQ", "fU"]
    else:
        log("Invalid method for RM fitting", stype = "err")
        return (None,) * 3


    # create figure to plot RM fits
    fig_RM, AX_RM = plt.subplots(*plotnum2grid(nrows = 5, num = ncomps), figsize = (10,12))
    AX_RM = AX_RM.flatten()


    #-------------#
    # FIT FOR RM  #
    #-------------#                  

    # loop over components and fit RM
    for i, t_crop in enumerate(tcrops):

        # update tcrop and fcrop
        par['t_crop'] = t_crop
        par['f_crop'] = fcrops[i]

        # get data
        _, _, _f, _freq = _proc_data(stk = stk, freq = freqs, data_list = data_list, 
                                par = par)

        if method == "RMquad":
            RM[i], RM_err[i], pa0[i], _ = fit_RMquad(Q = _f['Q'], U = _f['U'], Qerr = _f['Qerr'], 
                                                Uerr = _f['Uerr'], f = _freq, f0 = f0[i], **kwargs)

        elif method == "RMsynth":
            RM[i], RM_err[i], f0[i], pa0[i] = fit_RMsynth(I = _f['I'], Q = _f['Q'], U = _f['U'],
                                                    Ierr = _f['Ierr'], Qerr = _f['Qerr'], Uerr = _f['Uerr'],
                                                    f = _freq, **kwargs)
        
        log(f"Fitted RM: {RM[i]} +/- {RM_err[i]}\n")

        # plot RM fit
        plot_RM(Q = _f['Q'], U = _f['U'], Qerr = _f['Qerr'], Uerr = _f['Uerr'], 
                    f = _freq, rm = RM[i], pa0 = pa0[i], f0 = f0[i], ax = AX_RM[i])

        AX_RM[i].set(title = f"Component {i+1}")
    





    #---------------#
    # calc PA       #
    #---------------#

    # create figure
    fig_PA, AX_PA = plt.subplot_mosaic("P;S;D", figsize = (12, 10), 
            gridspec_kw={"height_ratios": [1, 2, 2]}, sharex=True)

    # data list
    data_list = ["dsI", "dsQ", "dsU", 
                       "tI",  "tQ",  "tU", 
                       "fQ",  "fU", "tV"]

    # loop over components and de-rotate then calc PA
    for i, t_crop in enumerate(tcrops):

        # update tcrop, fcrop, RM, pa0 and f0
        par['t_crop'] = t_crop
        par['f_crop'] = fcrops[i]
        par['RM'] = RM[i]
        par['pa0'] = 0.0
        par['f0'] = f0[i]

        # get data
        _ds, _t, _f, _ = _proc_data(stk = stk, freq = freqs, data_list = data_list, 
                                par = par)

        ## calculate PA
        stk_data = {"Q":_ds["Q"], "U":_ds["U"], "tQerr":_t["Qerr"],
                    "tUerr":_t["Uerr"], "tIerr":_t["Ierr"], "fQerr":_f["Qerr"],
                    "fUerr":_f["Uerr"]}
        PA_i, PA_err_i = calc_PAdebiased(stk_data, Ldebias_threshold = Ldebias_threshold)

        _x = np.linspace(t_crop[0]*tMAX, t_crop[1]*tMAX, PA_i.size)

        ## plot PA
        plot_PA(_x, PA_i, PA_err_i, ax = AX_PA['P'])

        ## plot stokes
        stk_t = {"I":_t['I'], "Q":_t['Q'], "U":_t['U'], "V":_t['V']}
        plot_stokes(_x, stk_t, stk_type = "t", ax = AX_PA['S'], L = plot_L)

    
    # plot full Stokes I dynamic spectrum
    full_tcrop = [tcrops[0][0], tcrops[-1][1]]
    full_fcrop = [0.0, 1.0]

    # update pars
    par['t_crop'] = full_tcrop
    par['f_crop'] = full_fcrop
    par['RM'] = 0.0

    _ds, _, _, _freq = _proc_data(stk = stk, freq = freqs, data_list = ["dsI"], 
                                par = par)
    
    t_lim = [full_tcrop[0]*tMAX, full_tcrop[1]*tMAX]
    f_lim = [_freq[-1], _freq[0]]

    AX_PA['D'].imshow(_ds['I'], aspect = 'auto', extent = [*t_lim, *f_lim])
    AX_PA['D'].set_ylabel("Frequency [MHz]", fontsize = 12)
    AX_PA['D'].set_xlabel("Time [ms]", fontsize = 12)

    # final figure params
    fig_PA.tight_layout()
    fig_PA.subplots_adjust(hspace = 0)
    AX_PA['P'].get_xaxis().set_visible(False)
    AX_PA['S'].get_xaxis().set_visible(False)
    AX_PA['S'].get_legend().remove()

    if plot_L:
        leg_s = ["I", "L", "V"]
    else:
        leg_s = ["I", "Q", "U", "V"]
    AX_PA['S'].legend(leg_s)




    if filename is None:
        plt.show()

    return (None,) * 3

