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
from .master_proc import _proc_par, master_proc_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .utils import plotnum2grid
from copy import deepcopy

# plotting params
default_col = plt.rcParams['axes.prop_cycle'].by_key()['color']


# NOTE:
# in the case of using the ilex interface and t_lim[0] is something other than 0.0,
# then this functions x axis in time won't be accurate

def multicomp_PA(stk, freqs, method = "RMquad", Ldebias_threshold = 2.0, dt = 0.001, plot_L = False,
                par: dict = None, tcrops = None, fcrops = None, filename: str = None, flipPA = False, ncols = 3,
                plot_err_type = "lines", **kwargs):
    """
    Multicomponent Polarisation, does full RM fitting and PA plotting/spectra plotting for each
    component specified by tcrops and fcrops

    Parameters
    ----------
    stk : dict
        Dictionary of memory maps for full STOKES IQUV HTR dynamic spectra \n
        [I] - Stokes I \n
        [Q] - Stokes Q \n
        [U] - Stokes U \n
        [V] - Stokes V
    freqs : ndarray
        Full Frequency array [MHz]
    method : str 
        Method for RM fitting \n
        [RMquad] - Use Quadratic function and Scipy.curve_fit \n
        [RMsynth] - Use RM synthesis from [RMtools] from CIRCADA [https://github.com/CIRADA-Tools/RM-Tools]
    Ldebias_threshold : float 
        Threshold for Debiased linear fraction in PA masking
    dt : float 
        Time resolution in [ms] of stk data
    plot_L: bool, optional
        Plot Linear Polarisation instead of Stokes Q and U, by default False
    tcrops : list
        List of Time phase limits for each component
    fcrops : list
        List of Freq phase limits for each component
    par : dict
        List of parameters for data processing, see master_proc_data() in master_proc() for a list of params
    plot_L : bool
        Plot Linear Fraction (L) instead of stokes Q and U
    filename : str 
        Save plots to files with a common prefix
    flipPA : bool, optional
        Plot PA from [0, 180] instead of [-90, 90], by default False
    ncols : number of colums when plotting a grid of different RM fitted components
    **kwargs: Dict
        Keyword arguments for fitting processes (see method for RM fitting)

    Returns:

    
    """
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
        
    
    log(f"Plotting PA for {len(tcrops)} components")

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
    fig_RM, AX_RM = plt.subplots(*plotnum2grid(ncols = ncols, num = ncomps), figsize = (14,14))
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
        _, _, _f, _freq = master_proc_data(stk = stk, freq = freqs, data_list = data_list, 
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
                    f = _freq, rm = RM[i], pa0 = pa0[i], f0 = f0[i], ax = AX_RM[i], 
                    plot_err_type = plot_err_type)

        AX_RM[i].set(title = f"Component {i+1}")
        AX_RM[i].set_ylim([-90, 90])
    
    for i, _ in enumerate(tcrops):
        AX_RM[i].set_xlabel("")
        AX_RM[i].set_ylabel("")

        if i % ncols == 0:
            AX_RM[i].set_ylabel("PA [deg]")
            
    # remove rest of unused axes
    gridx, gridy = plotnum2grid(ncols = ncols, num = ncomps)
        
    for i in range(ncomps - 1 - (ncomps-1)%ncols, ncomps):
        AX_RM[i].set_xlabel("Freq [MHz]")
    
    for i in range(len(tcrops),gridx*gridy):
        AX_RM[i].set_axis_off()

    





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
        _ds, _t, _f, _ = master_proc_data(stk = stk, freq = freqs, data_list = data_list, 
                                par = par)

        ## calculate PA
        stk_data = {"dsQ":_ds["Q"], "dsU":_ds["U"], "tQerr":_t["Qerr"],
                    "tUerr":_t["Uerr"], "tIerr":_t["Ierr"], "fQerr":_f["Qerr"],
                    "fUerr":_f["Uerr"]}

        PA_i, PA_err_i = calc_PAdebiased(stk_data, Ldebias_threshold = Ldebias_threshold)


        _x = np.linspace(t_crop[0]*tMAX, t_crop[1]*tMAX, PA_i.size)

        ## plot PA
        plot_PA(_x, PA_i, PA_err_i, ax = AX_PA['P'], flipPA = flipPA)

        ## plot stokes
        pdat = {'time':_x}
        for key in _t.keys():
            pdat[f"{key}"] = _t[key]
        plot_stokes(pdat, stk_type = "t", ax = AX_PA['S'], plot_L = plot_L, Ldebias = True,
                    plot_err_type = plot_err_type)

    
    # plot full Stokes I dynamic spectrum
    full_tcrop = [tcrops[0][0], tcrops[-1][1]]
    full_fcrop = [0.0, 1.0]

    # update pars
    par['t_crop'] = full_tcrop
    par['f_crop'] = full_fcrop
    par['RM'] = 0.0

    _ds, _, _, _freq = master_proc_data(stk = stk, freq = freqs, data_list = ["dsI"], 
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

    # make legend, use empty lineplots for legend
    
    if plot_L:
        leg_s = ["I", "L", "V"] 
        col_s = ["k", "r", "b"]
        leg_lines = [None] * 3 
    else:
        leg_s = ["I", "Q", "U", "V"]
        col_s = default_col[0:4]
        leg_lines = [None] * 4
    
    for i, lab in enumerate(leg_s):
        leg_lines[i], = AX_PA['S'].plot([],[],label = lab, color = col_s[i])
    AX_PA['S'].legend(leg_lines, leg_s)


    # print out infomation
    for i,_ in enumerate(RM):
        print(f"RM [{i}]= {RM[i]} +/- {RM_err[i]}   [rad/m2]")


    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return fig_PA


























# replace with using normal poincare plot multiple times
# def multicomp_poincare(stk, freqs, stk_type = "f", dt = 0.001, par: dict = None, 
#                         tcrops = None, fcrops = None, filename: str = None, sigma = 2.0, 
#                         plot_data = True, plot_model = False, plot_on_surface = True, 
#                         plot_P = False, n = 5, cbar_lims = [0.0, 1.0]):
#     """
#     Plot multiple tracks on Poincare sphere

#     Parameters
#     ----------
#     stk : dict
#         Dictionary of memory maps for full STOKES IQUV HTR dynamic spectra \n
#         [I] - Stokes I \n
#         [Q] - Stokes Q \n
#         [U] - Stokes U \n
#         [V] - Stokes V
#     freqs : ndarray
#         Full Frequency array [MHz]
#     stk_type : str, optional
#         stokes data to plot, by default "f" \n
#         [f] - plot stokes data as a function of frequency \n
#         [t] - plot stokes data as a function of time
#     dt : float, optional
#         time resolution in [ms], by default 0.001
#     tcrops : list
#         List of Time phase limits for each component
#     fcrops : list
#         List of Freq phase limits for each component
#     par : dict
#         List of parameters for data processing, see master_proc_data() in master_proc() for a list of params
#     filename : str, optional
#         filename to save plot to, by default None
#     sigma : float, optional
#         error threshold when masking stokes/I ratios, by default 2.0
#     plot_data : bool, optional
#         plot data on Poincare sphere, by default True
#     plot_model : bool, optional
#         Plot Polynomial fitted model of data on Poincare sphere, by default False
#     plot_on_surface : bool, optional
#         Plot data on surface of Poincare sphere (this will require normalising stokes data), by default True
#     plot_P : bool, optional
#         Plot Stokes/P instead of Stokes/I, by default False
#     n : int, optional
#         Maximum polynomial order for model fitting, by default 5
#     cbar_lims : list, optional
#         limits of colorbar, by default [0.0, 1.0]

#     Returns
#     -------
#     fig : figure
#         Return figure instance
#     """

#     # check if tcrops and fcrops are same length
#     if tcrops is None and fcrops is None:
#         log("Must specify time and/or freq crops for each component", stype = "err")
#         return (None,) * 3
#     elif tcrops is not None and fcrops is not None:
#         if len(tcrops) != len(fcrops):
#             log("number of time and freq crops must be equal", stype = "err")    
#             return (None,) * 3

#     # initialise par
#     par = _proc_par(par)

#     if par['terr_crop'] is None:
#         log("Off-pulse regions needs to be specified", stype = "err")

#     # initialise crops
#     if tcrops is None:
#         tcrops = [par['t_crop']] * len(fcrops)
        
#     if fcrops is None:
#         fcrops = [par['f_crop']] * len(tcrops)
        
        
#     log(f"Plotting poincare tracks for {len(tcrops)} components")


#     # initialise useful parameters
#     ncomps = len(tcrops)                    # number of components
#     tMAX = dt * stk['I'].shape[1]           # amount of time across htr data

#     # make plot with sphere
#     fig = plt.figure(figsize = (12,12))
#     ax = fig.add_subplot(111, projection = '3d')

#     # figure for plotting tracks on 2D 
#     fig2, ax2 = plt.subplots(1, 1, figsize = (10,10))
#     if stk_type == "t":
#         ax2.set(ylabel = "Flux (arb.)", xlabel = "Time [ms]")
#     elif stk_type == "f":
#         ax2.set(ylabel = "Flux (arb.)", xlabel = "Freq [MHz]")


#     def cart2sph(x, y, z):

#         # sgn(y)
#         sgny = np.zeros(y.size)
#         sgny[y < 0] = -1
#         sgny[y > 0] = 1

#         # r
#         r = np.sqrt(x**2 + y**2 + z **2)

#         # theta
#         theta = np.arccos(z/r)

#         # phi
#         phi = sgny * np.arccos(x/np.sqrt(x**2 + y**2))

#         return phi, theta

    
#     def sph2cart(r, phi, theta):

#         x = r * np.sin(phi) * np.cos(theta)
#         y = r * np.sin(phi) * np.sin(theta)
#         z = r * np.cos(phi)

#         return x, y, z


#     def set_axes_equal(ax: plt.Axes):
#         """Set 3D plot axes to equal scale.

#         Make axes of 3D plot have equal scale so that spheres appear as
#         spheres and cubes as cubes.  Required since `ax.axis('equal')`
#         and `ax.set_aspect('equal')` don't work on 3D.
#         """
#         limits = np.array([
#             ax.get_xlim3d(),
#             ax.get_ylim3d(),
#             ax.get_zlim3d(),
#         ])
#         origin = np.mean(limits, axis=1)
#         radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
#         _set_axes_radius(ax, origin, radius)

#     def _set_axes_radius(ax, origin, radius):
#         x, y, z = origin
#         ax.set_xlim3d([x - radius, x + radius])
#         ax.set_ylim3d([y - radius, y + radius])
#         ax.set_zlim3d([z - radius, z + radius])


#     # plot sphere surface
#     u = np.linspace(0, 2*np.pi, 200)
#     v = np.linspace(0, np.pi, 200)
#     u, v = np.meshgrid(u, v)
#     x = np.sin(u) * np.cos(v)
#     y = np.sin(u) * np.sin(v)
#     z = np.cos(u)
#     ax.plot_surface(x,y,z, color = [0.7, 0.7, 0.7, 0.3], shade = False)
#     ax.plot_wireframe(np.sin(u), np.sin(u)*0, np.cos(u), color = [0.4, 0.4, 0.4, 0.5], linestyle = '--')
#     ax.plot_wireframe(np.sin(u)*0, np.sin(u), np.cos(u), color = [0.4, 0.4, 0.4, 0.5], linestyle = '--')
#     ax.plot_wireframe(np.sin(u), np.cos(u), np.cos(u)*0, color = [0.4, 0.4, 0.4, 0.5], linestyle = '--')

#     # plot axes 
#     fig.tight_layout()
#     ax.plot([-1.0, 1.0], [0.0, 0.0], [0.0, 0.0], color = default_col[1], linestyle = '-.')
#     ax.plot([0.0, 0.0], [-1.0, 1.0], [0.0, 0.0], color = default_col[2], linestyle = '-.')
#     ax.plot([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0], color = default_col[3], linestyle = '-.')
#     ax.text(1.2, 0, 0, "Q", fontsize = 16, color = default_col[1])
#     ax.text(0, 1.2, 0, "U", fontsize = 16, color = default_col[2])
#     ax.text(0, 0, 1.2, "V", fontsize = 16, color = default_col[3])
#     ax.set_xlim([-1.2, 1.2])
#     ax.set_xlim([-1.2, 1.2])
#     ax.set_xlim([-1.2, 1.2])
#     ax.set_box_aspect([1,1,1])
#     set_axes_equal(ax)
#     ax.dist = 7.5
#     ax.set_axis_off()



#     # now loop through crops and plot tracks on sphere.
#     st = stk_type
#     data_list = [f"{st}I", f"{st}Q", f"{st}U", f"{st}V"]
#     for i, t_crop in enumerate(tcrops):

#         # update tcrop, fcrop, RM, pa0 and f0
#         par['t_crop'] = t_crop
#         par['f_crop'] = fcrops[i]

#         # get data
#         _, _t, _f, _freq = master_proc_data(stk = stk, freq = freqs, data_list = data_list, 
#                                 par = par) 
        
#         if stk_type == "t":
#             pdat = _t
#             xdat = np.linspace(t_crop[0]*tMAX, t_crop[1]*tMAX, pdat['I'].size)
#             _f = None
        
#         elif stk_type == "f":
#             pdat = _f
#             xdat = _freq
#             _t = None

#         # calculate Q/I, U/I, V/I
#         P = pdat['I'].copy()
#         Perr = pdat['Ierr'].copy()
#         if plot_P:
#             P = np.sqrt(pdat['Q']**2 + pdat['U']**2 + pdat['V']**2)
#             Perr = np.sqrt((pdat['Q']*pdat['Qerr'])**2 + 
#                         (pdat['U']*pdat['Uerr'])**2 + 
#                         (pdat['V']*pdat['Verr'])**2)/P
#         stk_mask = P >= sigma * Perr
#         stk_i = {}
        
#         for S in "QUV":
#             stk_i[S] = pdat[S][stk_mask]/P[stk_mask]
#         xdat = xdat[stk_mask]

#         stk_o = deepcopy(stk_i)


#         if plot_on_surface:
#             # get angles first
#             phi, theta = cart2sph(stk_i['Q'], stk_i['U'], stk_i['V'])

#             # imprint on surface of sphere
#             stk_i['Q'], stk_i['U'], stk_i['V'] = sph2cart(1.0, phi, theta)


#         # model stokes data
#         if plot_model:
#             stk_m = {}
#             stk_mo = {}
#             for S in "QUV":
#                 stk_m[S] = model_curve(stk_i[S], n = n, samp = 1000)
#                 stk_mo[S] = model_curve(stk_o[S], n = n, samp = 1000)

        
#         # plot data, we must be smart with how we plot these components.
#         # easiest just to take a single color gradient for each track
#         if plot_data:
#             alph_s, alph_w = 1.0, 0.7/xdat.size
#             for j in range(xdat.size):
#                 ax.plot(stk_i['Q'][j:j+2], stk_i['U'][j:j+2], stk_i['V'][j:j+2],
#                         color = default_col[i], alpha = alph_s - alph_w*j, linewidth = 3)
                
#         # plot model
#         if plot_model:
#             if plot_data:
#                 ax.plot(stk_m['Q'], stk_m['U'], stk_m['V'], color = 'r', 
#                         linestyle = '--', linewidth = 1.5)
#             else:
#                 alph_s, alph_w = 1.0, 0.7/stk_m['Q'].size
#                 for j in range(stk_m['Q'].size):
#                     ax.plot(stk_m['Q'][j:j+2], stk_m['U'][j:j+2], stk_m['V'][j:j+2], 
#                             color = default_col[i], alpha = alph_s - alph_w*j, linewidth = 3)
        

#         #-----------
#         # plot data on normal figure for diagnostic purposes
#         #-----------

#         # plot stokes 
#         markers = ["o","+","^"]
#         for k, S in enumerate("QUV"):
#             ax2.plot(xdat, stk_o[S], color = default_col[i], marker = markers[k], 
#                      linestyle = '')
#             if plot_model:
#                 ax2.plot(model_curve(xdat, n = 1, samp = 1000), stk_mo[S],
#                 color = default_col[i], linestyle = '--')
            
        
    
#     # make legend
#     leg_lines = [None] * ncomps
#     leg_lines2 = [None] * (ncomps + 3)
#     leg_names = [None] * ncomps
#     for i,_ in enumerate(tcrops):
#         leg_lines[i], = ax.plot([],[],[], color = default_col[i])
#         leg_lines2[i], = ax2.plot([],[], color = default_col[i])
#         leg_names[i] = f"Comp {i+1}"

#     leg_names2 = deepcopy(leg_names)
#     leg_lines2[i+1], = ax.plot([],[], color = 'k', marker = markers[0])
#     leg_names2 += ["Q/I"]
#     leg_lines2[i+2], = ax.plot([],[], color = 'k', marker = markers[1])
#     leg_names2 += ["U/I"]
#     leg_lines2[i+3], = ax.plot([],[], color = 'k', marker = markers[2])
#     leg_names2 += ["V/I"]
    
#     ax.legend(leg_lines, leg_names)
#     ax2.legend(leg_lines2, leg_names2)

#     # plot figure
#     if filename is not None:
#         plt.savefig(filename)
#     plt.show()
#     return fig

