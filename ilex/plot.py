##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 25/09/2023 
##
##
## 
## 
## Library of functions to plot data 
## 
## 
##
##===============================================##
##===============================================##
# imports
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

## import stats functions ##
from .fitting import (lorentz, scatt_pulse_profile, scat,
                       gaussian, model_curve)

from .globals import c, _G

from .data import *

from .utils import load_plotstyle, fix_ds_freq_lims

# constants
default_col = plt.rcParams['axes.prop_cycle'].by_key()['color']

#--------------------#
# to set up as global set func
#--------------------#
ILEX_PLOT_FONTSIZE = 16
ILEX_PLOT_ERRTYPE = "regions"



#-------------------------------------------------#
# UTILITY FUNCTIONS FOR PLOTTING                  #
#-------------------------------------------------#


def _data_from_dict(dic, keys):
    """
    Get data from dictionary

    """

    # check if data is there
    dic_keys = dic.keys()
    out_dic = {}

    # put data into output dictionary
    for key in keys:
        if key not in dic_keys:
            print("not all data given")
            return (None, ) * 2
        out_dic[key] = dic[key].copy()
    
    for key in ["freq", "time"]:
        if key in dic_keys:
            out_dic[key] = dic[key].copy()


    
    # now check if error data has been given
    err_flag = True
    err_keys = []
    for key in keys:
        if (key not in ["freq", "time"]) and ("err" not in key):
            err_keys += [f"{key}err"]
            if f"{key}err" in dic_keys:
                out_dic[f"{key}err"] = dic[f"{key}err"].copy()
            else:
                err_flag = False

    # if false, set all errors to false for convenience
    if not err_flag:
        for key in err_keys:
            out_dic[key] = None
    
    return out_dic, err_flag



# def _plot_err_as_lines(x, y, err, ax, col = 'k', linestyle = '', **kwargs):
#     ax.scatter(x, y, c = col, s = 6, **kwargs)
#     ax.errorbar(x, y, yerr = err, linestyle = linestyle, ecolor = col, 
#                 alpha = 0.5)

#     return 



# def _plot_err_as_regions(x, y, err, ax, col = 'k', **kwargs):
#     ax.plot(x, y, color = col, **kwargs)
#     ax.fill_between(x, y-err, y+err, color = col, alpha = 0.5,
#                     edgecolor = None)

#     return


# def _plot_err(x, y, err, ax, col = 'k', linestyle = '', plot_type = "lines", **kwargs):

#     if plot_type == "lines":
#         _plot_err_as_lines(x, y, err, ax, col = col, linestyle = linestyle, **kwargs)
    
#     elif plot_type == "regions":
#         _plot_err_as_regions(x, y, err, ax, col = col, **kwargs)
    
#     return












def _PLOT(x, y, yerr = None, ax = None, plot_type = "regions", color = None, alpha = 0.5,
            **kwargs):
    """
    General plotting function
    """

    if plot_type == "scatter":
        _PLOT_SCATTER(x = x, y = y, yerr = yerr, ax = ax, color = color, alpha = alpha, **kwargs)

    elif plot_type == "regions":
        _PLOT_REGIONS(x = x, y = y, yerr = yerr, ax = ax, color = color, alpha = alpha, **kwargs)

    else:
        print("Plot err style undefined/unsupported. ")
    
    return

def _PLOT_SCATTER(x, y, yerr = None, ax = None, color = None, alpha = 0.5, **kwargs):
    """
    Plot lines
    """

    plot_pars = load_plotstyle()

    for key in kwargs:
        if key in _G.scatter_args:
            plot_pars['scatter'][key] = kwargs[key]
            continue

        if key in _G.errorbar_args:
            plot_pars['errorbar'][key] = kwargs[key]

    # check if colors not in 
    for _p in ['c', 'facecolors']:
        if 'c' in plot_pars['scatter'].keys():
            del plot_pars['scatter']['c']
    
    for _p in ['ecolor', 'alpha', 'markerfacecolor']:
        if _p in plot_pars['errorbar'].keys():
            del plot_pars['errorbar'][_p]
    
    

    # plot scatter
    if ax is not None:
        sc = ax.scatter(x, y, c = color, facecolors = color, **plot_pars['scatter'])

        if yerr is not None:
            if color is None:
                color = sc.get_facecolors()[0]
            ax.errorbar(x = x, y = y, yerr = yerr, ecolor = color, 
                    alpha = alpha, markerfacecolor = color,
                         **plot_pars['errorbar'])

    else:
        sc = plt.scatter(x = x, y = y, c = color, facecolors = color, **plot_pars['scatter'])

        if yerr is not None:
            if color is None:
                color = sc.get_facecolors()[0]
            plt.errorbar(x, y, yerr = yerr, ecolor = color,
                    alpha = alpha, markerfacecolor = color, **plot_pars['errorbar'])

    return


def _PLOT_REGIONS(x, y, yerr = None, ax = None, color = None, alpha = 0.5, **kwargs):
    """
    Plot regions
    """

    plot_pars = load_plotstyle()

    for key in kwargs:
        if key in _G.plot_args:
            plot_pars['plot'][key] = kwargs[key]

    if 'color' in plot_pars['plot'].keys():
        del plot_pars['plot']['color']

    # plot region
    if ax is not None:
        ln, = ax.plot(x, y, color = color, **plot_pars['plot'])

        if yerr is not None:
            ax.fill_between(x, y-yerr, y+yerr, color = ln.get_color(), alpha = alpha,
                        edgecolor = None)

    else:
        ln, = plt.plot(x, y, color = color, **plot_pars['plot'])

        if yerr is not None:
            plt.fill_between(x, y-yerr, y+yerr, color = ln.get_color(), alpha = alpha,
                        edgecolor = None)
    
    return











def plot_data(dat, typ = "dsI", ax = None, filename: str = None, plot_type = "scatter"):
    """
    Plot data

    Parameters
    ----------
    dat : Dict(np.ndarray)
        Dictionary of stokes data, can include any data products
    typ : str, optional
        Type of data to plot, by default "dsI" \n
        [ds] - dynamic spectra \n
        [t] - time series \n
        [f] - frequency spectra
    ax : Axes, optional
        axes handle, by default None
    filename : str, optional
        filename to save figure to, by default None
    plot_type : str, optional
        type of plotting, by default "scatter"

    Returns
    -------
    fig : figure
        Return Figure instance
    """

    ##==================##
    ## PLOT START GUARD ##
    ##==================##
    fig_flag = True
    if ax is None:
        fig_flag = False
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])


    err_flag = False
    pdat, err_flag = _data_from_dict(dat, list([typ]))
    if pdat is None:
        return


    # check if freq array was given, else make phase array
    if "freq" not in pdat.keys():
        fname = "Freq (phase)"
        flim = [0.0, 1.0]
    else:
        fname = "Freq [MHz]"
        df = abs(pdat['freq'][1] - pdat['freq'][0])
        flim = [np.min(pdat['freq']), np.max(pdat['freq'])] # for ds plotting
        flim = fix_ds_freq_lims(flim, df)
    
    # check if time array was given, else make phase array
    if "time" not in pdat.keys():
        tname = "Time (phase)"
        tlim = [0.0, 1.0]    
    else:
        tname = "Time [ms]"
        tlim = [pdat['time'][0], pdat['time'][-1]]

    
    # utility functions
    def plot_freq(x, y):
        ax.plot(x, y, 'k')
        ax.set(xlabel = fname, ylabel = "Flux Density")


    # check type 
    if typ[0:2] == "ds":
        # plot dynspec
        ax.imshow(pdat[typ], aspect = 'auto', extent = [*tlim, *flim])
        ax.set(xlabel = tname, ylabel = fname)
    
    elif typ[0] == "t":
        # scrunch in freq
        tx = np.linspace(*tlim, pdat[typ].size)
        ax.set(xlabel = tname, ylabel = "Flux Density (arb.)")
        _PLOT(tx, pdat[typ], pdat[f"{typ}err"], ax = ax, color = 'k', alpha = 0.5,
                        plot_type = plot_type)

    elif typ[0] == "f":
        # scrunch in time
        fx = np.linspace(*flim, pdat[typ].size)
        ax.set(xlabel = fname, ylabel = "Flux Density (arb.)")
        _PLOT(fx, pdat[typ], pdat[f"{typ}err"], ax = ax, color = 'k', alpha = 0.5,
                        plot_type = plot_type)

    else:
        print("Invalid data type to plot")


    ##================##
    ## PLOT END GUARD ##
    ##================##
    if not fig_flag:
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        return fig
    
    return None





def plot_RM(f, Q, U, Qerr = None, Uerr = None, rm = 0.0, pa0 = 0.0, f0 = 0.0,
            ax = None, filename: str = None, plot_type = "scatter"):
    """
    Plot RM fit

    Parameters
    ----------
    f : np.ndarray
        Frequency array
    Q : np.ndarray
        Stokes Q spectrum
    U : np.ndarray
        Stokes U spectrum
    Qerr : np.ndarray, optional
        Stokes Q error spectrum, by default None
    Uerr : np.ndarray, optional
        Stokes U error spectrum, by default None
    rm : float, optional
        Rotation Measure [rad/m^2], by default 0.0
    pa0 : float, optional
        position angle at f0, by default 0.0
    f0 : float, optional
        reference frequency [MHz], by default 0.0
    ax : Axes, optional
        Axes handle, by default None
    filename : str, optional
        filename to save figure to, by default None
    plot_type : str, optional
        type of error to plot, by default "scatter"

    Returns
    -------
    fig : figure
        Return figure instance
    """

    ##==================##
    ## PLOT START GUARD ##
    ##==================##
    fig_flag = True
    if ax is None:
        fig_flag = False
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    err_flag = (Uerr is not None and Qerr is not None)



    def rmquad(f, rm, pa0):
        angs = pa0 + rm*c**2/1e12*(1/f**2 - 1/f0**2)
        return 0.5*np.arctan2(np.sin(2*angs), np.cos(2*angs))


    # set up axes
    ax.set_xlabel("Frequency [MHz]", fontsize = 12)
    ax.set_ylabel("PA [deg]", fontsize = 12)

    # calc PA
    # PA = 0.5 * np.arctan2(U, Q)
    PA, PAerr = calc_PA(Q, U, Qerr, Uerr)
    PA_fit = rmquad(f, rm, pa0)

    _PLOT(x = f, y = PA*180/np.pi, yerr = PAerr*180/np.pi, ax = ax, color = 'k', alpha = 0.5,
            plot_type = plot_type)

    # # plot 
    # # ax.scatter(f, PA * 180/np.pi, c = 'k', s = 5, label  = "Measured PA")                # PA from data
    # # ax.plot(f, PA_fit * 180/np.pi, 'r', label = f"RM: {rm:.3f},    pa0: {pa0:.3f}")      # PA best fit line
    
    # # error plotting
    # if err_flag:
    #     _, PA_err = calc_PA(Q, U, Qerr, Uerr)
        
    #     _plot_err(f, PA * 180/np.pi, PA_err * 180/np.pi, ax = ax, col = [0., 0., 0., 0.5], 
    #     plot_type = plot_err_type)
    
    ax.set_ylim([-90, 90])

    ax.legend()





    ##================##
    ## PLOT END GUARD ##
    ##================##
    if not fig_flag:
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        return fig
    
    return None













def plot_PA(x, PA, PA_err, ax = None, flipPA = False, filename: str = None,
            plot_type = "scatter"):
    """
    Plot PA profile

    Parameters
    ----------
    x : np.ndarray
        X data
    PA : np.ndarray
        Position angle
    PA_err : np.ndarray
        PA error
    ax : Axes, optional
        Axes handle, by default None
    flipPA : bool, optional
        plot PA over [0, 180] degrees instead of [-90, 90], by default False
    filename : str, optional
        filename to save figure to, by default None
    plot_type : str, optional
        type of error to plot, by default "scatter"

    Returns
    -------
    fig : figure
        Return figure instance
    """

    ##==================##
    ## PLOT START GUARD ##
    ##==================##
    fig_flag = True
    if ax is None:
        fig_flag = False
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])




    # make axes
    ax.set_ylabel("PA [deg]", fontsize = 12)
    ax.set_xlabel("time [ms]", fontsize = 12)


    # flip PA
    if flipPA:
        PA[PA < 0] += np.pi
    
    # plot PA
    # PA_mask = ~np.isnan(PA)
    # ax.scatter(x, PA * 180/np.pi, c = 'k', s = 2)
    _PLOT(x = x, y = PA * 180/np.pi, yerr = PA_err * 180/np.pi, color = 'k',
                alpha = 0.5, plot_type = plot_type, ax = ax)

    if flipPA:
        ax.set_ylim([0, 180])
    else:
        ax.set_ylim([-90, 90])




    ##================##
    ## PLOT END GUARD ##
    ##================##
    if not fig_flag:
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        return fig
    
    return None





def plot_stokes(dat, Ldebias = False, sigma = 2.0, stk_type = "f", stk2plot = "IQUV", 
                stk_ratio = False, ax = None, filename: str = None, plot_type = "scatter"):
    """
    Plot Stokes data, by default stokes I, Q, U and V data is plotted

    Parameters
    ----------
    dat : Dict(np.ndarray)
        Dictionary of stokes data, can include any data products but must include the following: \n
        [<x>I] - Stokes I data \n
        [<x>Q] - Stokes Q data \n
        [<x>U] - Stokes U data \n
        [<x>V] - Stokes V data \n
        [<x>Ierr] - Stokes I error data, only if Ldebias = True or stk_ratio = True \n
        where <x> is either 't' for time and 'f' for freq
    Ldebias : bool, optional
        Plot stokes L debias, by default False
    sigma : float, optional
        sigma threshold for error masking, I < sigma * Ierr, mask it out or
        else weird overflow behavior might be present when calculating stokes ratios, by default 2.0
    stk_type : str, optional
        Type of stokes data to plot, "f" for Stokes Frequency data or "t" for time data, by default "f"
    stk2plot : str, optional
        string of stokes to plot, for example if "QV", only stokes Q and V are plotted, by default "IQUV", choice between "IQUVLP"
    stk_ratio : bool, optional
        if true, plot stokes ratios S/I
    filename : str, optional
        name of file to save figure image, by default None
    plot_type : str, optional
        Choose between two methods of plotting the error in the data, by default "scatter" \n
        [regions] - Show error in data as shaded regions
        [scatter] - Show error in data as tics in markers

    Returns
    -------
    fig : figure
        Return figure instance

    """

    ##==================##
    ## PLOT START GUARD ##
    ##==================##
    fig_flag = True
    if ax is None:
        fig_flag = False
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])


    ## check if frequency or time data
    if stk_type == "t":
        ax.set_xlabel("Time [ms]", fontsize = 12)
        xdat = "time"
    elif stk_type == "f":
        ax.set_xlabel("Frequency [MHz]", fontsize = 12)
        xdat = "freq"
    else:
        print("Invalid type")
    ## update ax
    ax.set_ylabel("Flux Density. ", fontsize = 12)

    st = stk_type

    data_list = [f"{st}I", f"{st}Q", f"{st}U", f"{st}V", xdat]
    if Ldebias:
        data_list += [f"{st}Ierr"]

    # get data
    pdat, err_flag = _data_from_dict(dat, data_list)

    # stokes to plot
    col = default_col[0:4]
    col = {"I":'k', "Q":col[1], "U":col[2], "V":'b', "L": 'r', "P": 'darkviolet'}

    # check if L was given in plotting
    if "L" in stk2plot:

        if Ldebias:
            pdat[f"{st}L"], pdat[f"{st}Lerr"] = calc_Ldebiased(pdat[f"{st}Q"], pdat[f"{st}U"],
                                        pdat[f'{st}Ierr'], pdat[f'{st}Qerr'], pdat[f'{st}Uerr'])
        else:
            pdat[f"{st}L"], pdat[f"{st}Lerr"] = calc_L(pdat[f'{st}Q'], pdat[f'{st}U'], pdat[f'{st}Qerr'],
                                        pdat[f'{st}Uerr'])

    # check if P was given in plotting
    if "P" in stk2plot:

        if Ldebias:
            pdat[f"{st}P"], pdat[f"{st}Perr"] = calc_Pdebiased(pdat[f"{st}Q"], pdat[f"{st}U"], pdat[f"{st}V"],
                                        pdat[f'{st}Ierr'], pdat[f'{st}Qerr'], pdat[f'{st}Uerr'], pdat[f'{st}Verr'])
        else:
            pdat[f"{st}P"], pdat[f"{st}Perr"] = calc_P(pdat[f'{st}Q'], pdat[f'{st}U'], pdat[f'{st}V'],
                                                       pdat[f'{st}Qerr'], pdat[f'{st}Uerr'], pdat[f'{st}Verr'])

    
    # plot stokes ratios
    if stk_ratio:
        stk2plot = stk2plot.replace("I", "") # remove part of string

        # get sigma mask
        sigma_mask = pdat[f"{st}I"] < sigma * pdat[f"{st}Ierr"]

        # calc ratios
        for S in stk2plot:
            pdat[f"{st}{S}"], pdat[f"{st}{S}err"] = calc_ratio(pdat[f"{st}I"], pdat[f"{st}{S}"],
                                                        pdat[f"{st}Ierr"], pdat[f"{st}{S}err"], keep_size = False)

            # mask values with too large errors
            pdat[f"{st}{S}"][sigma_mask] = np.nan
            pdat[f"{st}{S}err"][sigma_mask] = np.nan




    # now we are ready to plot stokes data
    for i, S in enumerate(stk2plot):
        _PLOT(x = pdat[xdat], y = pdat[f"{st}{S}"], yerr = pdat[f"{st}{S}err"], ax = ax, color = col[S],
             plot_type = plot_type, label = S)
    
    ax.legend()



    ##================##
    ## PLOT END GUARD ##
    ##================##
    if not fig_flag:
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        return fig
    
    return None





def create_poincare_sphere(cbar_lims, cbar_label):
    """
    Create poincare sphere plot
    
    Parameters
    ----------
    cbar_lims : list(float)
        colorbar limits
    cbar_label : str
        colorbar label 

    Returns
    -------
    fig : figure
        figure instance
    
    """
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111, projection = '3d')


    def set_axes_equal(ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        _set_axes_radius(ax, origin, radius)

    def _set_axes_radius(ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])


    # plot sphere surface
    u = np.linspace(0, 2*np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    u, v = np.meshgrid(u, v)
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    ax.plot_surface(x,y,z, color = [0.7, 0.7, 0.7, 0.3], shade = False)
    ax.plot_wireframe(np.sin(u), np.sin(u)*0, np.cos(u), color = [0.4, 0.4, 0.4, 0.5], linestyle = '--')
    ax.plot_wireframe(np.sin(u)*0, np.sin(u), np.cos(u), color = [0.4, 0.4, 0.4, 0.5], linestyle = '--')
    ax.plot_wireframe(np.sin(u), np.cos(u), np.cos(u)*0, color = [0.4, 0.4, 0.4, 0.5], linestyle = '--')

    # plot axes 
    fig.tight_layout()
    ax.plot([-1.0, 1.0], [0.0, 0.0], [0.0, 0.0], color = default_col[1], linestyle = '-.')
    ax.plot([0.0, 0.0], [-1.0, 1.0], [0.0, 0.0], color = default_col[2], linestyle = '-.')
    ax.plot([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0], color = default_col[3], linestyle = '-.')
    ax.text(1.2, 0, 0, "Q", fontsize = 16, color = default_col[1])
    ax.text(0, 1.2, 0, "U", fontsize = 16, color = default_col[2])
    ax.text(0, 0, 1.2, "V", fontsize = 16, color = default_col[3])
    ax.set_xlim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    ax.dist = 7.5
    ax.set_axis_off()


    # create colorbar
    ax_c = fig.add_axes([0.2, 0.07, 0.6, 0.02])
    ax_c.get_yaxis().set_visible(False)
    ax_c.imshow(np.linspace(0,1.0, 256).reshape(1, 256)[::-1], aspect = 'auto', extent = [*cbar_lims, 0.0, 1.0],
                cmap = 'viridis')
    ax_c.set_xlabel(cbar_label)

    return fig, ax












# split into two functions, one to plot sphere, other to plot track in 3D
def plot_poincare_track(dat, ax, sigma = 2.0, plot_data = True, plot_model = False,
                    normalise = True, n = 5, filename: str = None):
    """
    Plot Stokes data on a Poincare Sphere.

    Parameters
    ----------
    dat : Dict(np.ndarray)
        Dictionary of stokes data, can include any data products but must include the following: \n
        [I] - Stokes I data \n
        [Q] - Stokes Q data \n
        [U] - Stokes U data \n
        [V] - Stokes V data \n
        [Ierr] - Stokes I error data
    filename : str, optional
        filename to save figure to, by default None
    stk_type : str, optional
        types of stokes data to plot, by default "f" \n
        [f] - Plot as a function of frequency \n
        [t] - Plot as a function of time 
    sigma : float, optional
        Error threshold used for masking stokes data in the case that stokes/I is being calculated \n
        this avoids deviding by potentially small numbers and getting weird results,by default 2.0
    plot_data : bool, optional
        Plot Data on Poincare sphere, by default True
    plot_model : bool, optional
        Plot Polynomial fitted data on Poincare sphere, by default False
    normalise : bool, optional
        Plot data on surface of Poincare sphere (this will require normalising stokes data), by default True
    n : int, optional
        Maximum order of Polynomial fit, by default 5

    Returns
    -------
    stk_i: Stokes 1D arrays
    stk_m: Stokes model 1D arrays
    """    

    data_list = ["I", "Q", "U", "V", "Ierr"]

    # get data
    pdat, err_flag = _data_from_dict(dat, data_list)

    # calculate stokes ratios
    # choice of normalizing against stokes I or P
    P = pdat['I'].copy()
    Perr = pdat['Ierr'].copy()

    if not err_flag:
        print("stk/P requires all stokes err")
        normalise = False
    if normalise:
        P, Perr = calc_Pdebiased(pdat['Q'], pdat['U'], pdat['V'], pdat['Ierr'],
                                pdat['Qerr'], pdat['Uerr'], pdat['Verr'])
    stk_mask = P >= sigma * Perr
    stk_i = {}
    
    for S in "QUV":
        stk_i[S] = pdat[S].copy()
        stk_i[S][stk_mask] = pdat[S][stk_mask]/P[stk_mask]
        stk_i[S][~stk_mask] = np.nan
    stk_o = deepcopy(stk_i)

    # model stokes data
    if plot_model:
        stk_m = {}
        stk_mo = {}
        for S in "QUV":
            stk_m[S] = model_curve(stk_i[S], n = n, samp = 1000)
            # stk_mo[S] = model_curve(stk_o[S], n = n, samp = 1000)

    # plot stokes data
    if plot_data:
        cols = cm.viridis(np.linspace(0, 1, stk_i['Q'].size - 1))
        for i in range(stk_i['Q'].size - 1):
            ax.plot(stk_i['Q'][i:i+2], stk_i['U'][i:i+2], stk_i['V'][i:i+2], color = cols[i], linewidth = 3)

    # plot model
    if plot_model:
        if plot_data:
            ax.plot(stk_m['Q'], stk_m['U'], stk_m['V'], color = 'r', linestyle = '--', linewidth = 1.5)
        else:
            cols = cm.viridis(np.linspace(0, 1, stk_m['Q'].size - 1))
            for i in range(stk_m['Q'].size - 1):
                ax.plot(stk_m['Q'][i:i+2], stk_m['U'][i:i+2], stk_m['V'][i:i+2], color = cols[i], linewidth = 3)

    
    return stk_i, stk_m
    