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

from .data import *
# constants
c = 2.997924538e8 # Speed of light [m/s]
default_col = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_help():
    help_path = os.path.join(os.environ['ILEX_PATH'],"help/plot_help.txt")
    with open(help_path, 'r') as f:
        help_string = f.read()

    print(help_string)

    return


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



def _plot_err_as_lines(x, y, err, ax, col = 'k', linestyle = ''):

    ax.errorbar(x, y, yerr = err, linestyle = linestyle, ecolor = col, 
                alpha = 0.5)

    return 



def _plot_err_as_regions(x, y, err, ax, col = 'k'):
    ax.plot(x, y, color = col)
    ax.fill_between(x, y-err, y+err, color = col, alpha = 0.5,
                    edgecolor = None)

    return


def _plot_err(x, y, err, ax, col = 'k', linestyle = '', plot_type = "lines"):

    if plot_type == "lines":
        _plot_err_as_lines(x, y, err, ax, col = col, linestyle = linestyle)
    
    elif plot_type == "regions":
        _plot_err_as_regions(x, y, err, ax, col = col)
    
    return















def plot_data(dat, typ = "dsI", ax = None, filename: str = None, plot_err_type = "lines"):
    """
    Info:
        Plot data
    
    Args:
        data (ndarray): Stokes Dynamic spectrum
        typ (str): type of data to plot
                    [ds] -> Dynamic spectra
                    [t] -> Time series
                    [f] -> Spectra
        tlim (List): Time bounds (limits) in [ms]
        flim (List): Freq bounds (limits) in [MHz]
        ax (axes): Axes handle
        filename (str): To save figure to

    
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
        flim = [pdat['freq'][0], pdat['freq'][-1]]
    
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
        ax.set(xlabel = fnname, ylabel = "Flux Density")


    # check type 
    if typ[0:2] == "ds":
        # plot dynspec
        ax.imshow(pdat[typ], aspect = 'auto', extent = [*tlim, *flim[::-1]])
        ax.set(xlabel = tname, ylabel = fname)
    
    elif typ[0] == "t":
        # scrunch in freq
        tx = np.linspace(*tlim, pdat[typ].size)
        ax.plot(tx, pdat[typ], 'k')
        ax.set(xlabel = tname, ylabel = "Flux Density (arb.)")

        if err_flag:
            _plot_err(tx, pdat[typ], pdat[f"{typ}err"], ax = ax, col = [0., 0., 0., 0.5],
                        plot_type = plot_err_type)

    elif typ[0] == "f":
        # scrunch in time
        fx = np.linspace(*flim, pdat[typ].size)
        ax.plot(fx, pdat[typ], 'k')
        ax.set(xlabel = fname, ylabel = "Flux Density (arb.)")

        if err_flag:
            _plot_err(fx, pdat[typ], pdat[f"{typ}err"], ax = ax, col = [0., 0., 0., 0.5],
                        plot_type = plot_err_type)

    else:
        print("Invalid data type to plot")


    ##================##
    ## PLOT END GUARD ##
    ##================##
    if filename is not None and fig_flag:
        plt.savefig(filename)
    elif not fig_flag:
        plt.show()
        return fig

    return None






## [ PLOT SCINTILLATION ] ##
# TODO: change args [p] for [w, a] par types
def plot_scintband(x, y, y_err = None, w = 0.0, a = 0.0, ax = None, filename: str = None,
                    plot_err_type = "lines"):
    """
    Info:
        plot scintillation bandwidth

    Args:
        x (ndarray): x data
        y (ndarray): y data
        y_err (ndarray): y data error
        w (float): scintillation bandwidth
        a (float): amplitude
        filename (str): filename to save figure to

    Returns:
        fig (figure): figure handle


    """
    
    ##==================##
    ## PLOT START GUARD ##
    ##==================##
    fig_flag = True
    if ax is None:
        fig_flag = False
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])


    
    # start setting up axes
    ax.set_xlabel("Frequency Lag [Mhz]", fontsize = 12)
    ax.set_ylabel("normalised acf", fontsize = 12)

    # plot data
    ax.scatter(x, y, c = 'k', s = 1.5)

    # plot error bars if applicable
    if y_err is not None:
        # error bars
        _plot_err(x, y, yerr = y_err, ax = ax, col = [0., 0., 0., 0.5], plot_type = plot_err_type)

    # plot model
    ax.plot(np.linspace(0,x[-1],x.size), lorentz(x, w, a))

    # write decorrelation bandwidth in legend title
    leg_title = f"$\\nu_{{dc}} = {{{w:.4f}}}$ MHz"
    ax.legend(title = leg_title, fontsize = 14)




    ##================##
    ## PLOT END GUARD ##
    ##================##
    if filename is not None and fig_flag:
        plt.savefig(filename)
    elif not fig_flag:
        plt.show()
        return fig

    return None


    






## [ PLOT SCATTERING TIMESCALE ] ##
def plot_tscatt(x, y, y_err = None, ax = None, p = None, npulse: int = 1, filename: str = None,
                plot_err_type = "lines"):
    """
    Info:
        plot scattering timescale

    Args:
        x (ndarray): x data
        y (ndarray): y data
        y_err (ndarray): y data error
        ax (axes): Axes handle
        npulse (int): Number of convolved pulses from fit
        p (dict): parameters to pass, derived from get_tscatt()
        filename (str): filename to save figure to

    Returns:
        fig (figure): figure handle
    """

    ##==================##
    ## PLOT START GUARD ##
    ##==================##
    fig_flag = True
    if ax is None:
        fig_flag = False
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])






    # add to figure
    ax.set_xlabel("Time [ms]", fontsize = 12)
    ax.set_ylabel("normalised arb.", fontsize = 12)

    # plot data
    ax.scatter(x, y, c = 'k', s = 1.5)
    
    # plot error bars if applicable
    if y_err is not None:
        # error bars
        _plot_err(x, y, yerr = y_err, ax = ax, col = [0., 0., 0., 0.5], plot_type = plot_err_type)


    # plot model
    pvals = p.get_bestfit()
    ax.plot(x, scatt_pulse_profile(x,pvals))

    # plot scattered gaussian models
    tail_s = scat(x, pvals['tau'])
    for i in range(1,npulse+1):
        y_modl = np.convolve(gaussian(x, 1, pvals[f"mu{i}"], pvals[f"sig{i}"]),
                             tail_s, mode = "same")
        ax.plot(x, pvals[f"a{i}"] * y_modl/np.max(y_modl), '--')


    # make legend to show scattering timescale
    leg_title = f"$\\tau_{{s}} = {{{p.val['tau']:.4f}}}_{{-{p.plus['tau']:.4f}}}^{{+{p.minus['tau']:.4f}}}$ ms"
    ax.legend(title = leg_title, fontsize = 12)

    


    

    ##================##
    ## PLOT END GUARD ##
    ##================##
    if filename is not None and fig_flag:
        plt.savefig(filename)
    elif not fig_flag:
        plt.show()
        return fig

    return None










def plot_RM(f, Q, U, Qerr = None, Uerr = None, rm = 0.0, pa0 = 0.0, f0 = 0.0,
            ax = None, filename: str = None, plot_err_type = "lines"):
    """
    Info:
        Plot RM against PA spectra

    Args:
        Q (ndarray): stokes Q spectra
        U (ndarray): stokes U spectra
        f (ndarray): frequency array
        rm (float): rotation measure
        pa0 (float): position angle
        f0 (float): reference frequency
        ax (axes): axes handle
        filename (str): name to save figure to, if None figure is 
                        not saved.

    Returns
        fig: returns figure handle if ax handle not passed and 
             filename not given

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
    PA = 0.5 * np.arctan2(U, Q)
    PA_fit = rmquad(f, rm, pa0)

    # plot 
    ax.scatter(f, PA * 180/np.pi, c = 'k', s = 5, label  = "Measured PA")                # PA from data
    ax.plot(f, PA_fit * 180/np.pi, 'r', label = f"RM: {rm:.3f},    pa0: {pa0:.3f}")      # PA best fit line
    
    

    # error plotting
    if err_flag:
        _, PA_err = calc_PA(Q, U, Qerr, Uerr)
        
        _plot_err(f, PA * 180/np.pi, PA_err * 180/np.pi, ax = ax, col = [0., 0., 0., 0.5], 
        plot_type = plot_err_type)
    
    ax.set_ylim([-90, 90])

    ax.legend()





    ##================##
    ## PLOT END GUARD ##
    ##================##
    if filename is not None and fig_flag:
        plt.savefig(filename)
    elif not fig_flag:
        plt.show()
        return fig

    return None













def plot_PA(x, PA, PA_err, ax = None, flipPA = False, filename: str = None,
            plot_err_type = "lines"):
    """
    Info:
        Plot Position angle and error as function of time in units degrees

    Args:
        x (ndarray): x axis to plot PA units in [ms]
        PA (ndarray): position angle
        PA_err (ndarray): position angle error
        ax (axes): axes handle
        filename (str): filename of saved figure

    Returns
        fig (figure): returns figure handle if ax handle not passed and 
                      filename not given

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
    ax.scatter(x, PA * 180/np.pi, c = 'k', s = 2)
    _plot_err(x, PA * 180/np.pi, PA_err * 180/np.pi, col = 'k',
                plot_type = plot_err_type, ax = ax)

    if flipPA:
        ax.set_ylim([0, 180])
    else:
        ax.set_ylim([-90, 90])




    ##================##
    ## PLOT END GUARD ##
    ##================##
    if filename is not None and fig_flag:
        plt.savefig(filename)
    elif not fig_flag:
        plt.show()
        return fig

    return None














def plot_stokes(dat, plot_L = False, Ldebias = False, debias_threshold = 2.0, 
            stk_type = "f", stk_ratio = False, stk2plot = "IQUV", ax = None, filename: str = None,
            plot_err_type = "lines"):
    """
    Info:
        Plot Stokes time series/spectra

    Args:
        x (ndarray): x axis data, either time/frequency
        stk_data (dict): dictionary of stokes data
                         [I] -> Stokes I time/frequency data
                         [Q] -> Stokes Q time/frequency data
                         [U] -> Stokes U time/frequency data
                         [V] -> Stokes V time/frequency data
                         [Ierr] -> (Optional) for calculating L debiased
        stk_err (dict): dictionary of stokes errors
        lims (list): limits of x axis, either time in [ms] or freq in [MHz]
                     be treated as spectra, else if not as time 
                     series
        L (bool): If True, plots the Linear pol data 'L' instead
                  of Q and U
        ax (axes): axes handle
        stk_type (str): Time series "t" or spectra "f"
        filename (str): name of figure file to save

    Returns:
        fig: if filename and ax not specified, returns figure
             handle.

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

    data_list = ["I", "Q", "U", "V", xdat]
    if Ldebias or stk_ratio:
        data_list += [f"Ierr"]

    # get data
    pdat, err_flag = _data_from_dict(dat, data_list)

    # stokes to plot
    col = default_col[0:4]
    col = {"I":col[0], "Q":col[1], "U":col[2], "V":col[3]}
    if plot_L:
        col = {"I":'k', "L":'r', "V":'b'}

        # calc L
        if Ldebias:
            pdat["L"] = calc_Ldebiased(pdat["Q"], pdat["U"], pdat["Ierr"])
        else:
            pdat["L"] = calc_L(pdat["Q"], pdat["U"])
        
        if err_flag:
            pdat["Lerr"] = np.sqrt(pdat["Q"]**2*pdat["Qerr"]**2 + 
                                   pdat["U"]**2*pdat["Uerr"]**2)
            Lmask = pdat["L"] != 0.0
            pdat["Lerr"] = np.nanmean(pdat["Lerr"][Lmask]/pdat["L"][Lmask])

        # update stk2plot
        stk = ""
        if "I" in stk2plot:
            stk += "I"
        stk += "L"
        if "V" in stk2plot:
            stk += "V"
        stk2plot = stk


    # check if stk ratios being used
    linestyle = '-'
    marker = None
    if stk_ratio:
        # get mask for ratios
        stk_mask = pdat["I"] >= debias_threshold*pdat["Ierr"]

        # this is done assuming freq errors are arrays and time errors
        # are scalars
        Ierr = pdat["Ierr"]

        for S in stk2plot:
            if S != "I":
                Xerr = None
                if err_flag:
                    Xerr = pdat[f"{S}err"]
                pdat[f"{S}"], pdat[f"{S}err"] = calc_ratio(
                    pdat[f"I"], pdat[f"{S}"], Ierr, Xerr)
        
        # normalize I and Ierr, for completeness
        Iratio = pdat[f"I"]/pdat[f"I"]

        # rough calculation
        if st == "t":
            pdat[f"Ierr"] *= np.nanstd(1/pdat[f"I"][stk_mask])
        else:
            pdat[f"Ierr"] /= pdat[f"I"]
        pdat[f"I"] = Iratio

        # now time/freq array
        pdat[xdat][~stk_mask] = np.nan

        print(pdat)
        for S in stk2plot:
            pdat[f"{S}"][~stk_mask] = np.nan

            if st == "f" and err_flag:
                print(pdat[f"{S}err"])
                pdat[f"{S}err"][~stk_mask] = np.nan


        # set additional plotting parameters
        linestyle = '' # since masked stokes data may be segmented, we want to avoid drawing a line
        marker = '.'
        
    # now we are ready to plot stokes data
    for i, S in enumerate(stk2plot):
        ax.plot(pdat[xdat], pdat[f"{S}"], color = col[S], label = S, marker = marker, linestyle = linestyle,
                markersize = 5)
        if err_flag:
            _plot_err(pdat[xdat], pdat[f"{S}"], pdat[f"{S}err"], ax = ax, col= col[S],
             plot_type = plot_err_type)
    
    ax.legend()



    ##================##
    ## PLOT END GUARD ##
    ##================##
    if filename is not None and fig_flag:
        plt.savefig(filename)
    elif not fig_flag:
        plt.show()
        return fig

    return None 












def plot_poincare(dat, sigma = 2.0, plot_data = True, plot_model = False, 
                    plot_on_surface = True, n = 5, filename: str = None, cbar_lims = [0.0, 1.0], cbar_label = "", plot_P = False):
    """
    Info:
        Plot RM against PA spectra

    Args:
        Q (ndarray): stokes Q spectra
        U (ndarray): stokes U spectra
        f (ndarray): frequency array
        rm (float): rotation measure
        pa0 (float): position angle
        f0 (float): reference frequency
        ax (axes): axes handle
        filename (str): name to save figure to, if None figure is 
                        not saved.

    Returns
        fig: returns figure handle if ax handle not passed and 
             filename not given

    """

    data_list = ["I", "Q", "U", "V", "Ierr"]

    # get data
    pdat, err_flag = _data_from_dict(dat, data_list)



    ##==================##
    ## PLOT START GUARD ##
    ##==================##
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111, projection = '3d')


    def cart2sph(x, y, z):

        # sgn(y)
        sgny = np.zeros(y.size)
        sgny[y < 0] = -1
        sgny[y > 0] = 1

        # r
        r = np.sqrt(x**2 + y**2 + z **2)

        # theta
        theta = np.arccos(z/r)

        # phi
        phi = sgny * np.arccos(x/np.sqrt(x**2 + y**2))

        return phi, theta

    
    def sph2cart(r, phi, theta):

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        return x, y, z



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


    # calculate stokes ratios
    # choice of normalizing against stokes I or P
    P = pdat['I'].copy()
    Perr = pdat['Ierr'].copy()

    if not err_flag:
        print("stk/P requires all stokes err")
        plot_P = False
    if plot_P:
        P = np.sqrt(pdat['Q']**2 + pdat['U']**2 + pdat['V']**2)
        Perr = np.sqrt((pdat['Q']*pdat['Qerr'])**2 + 
                       (pdat['U']*pdat['Uerr'])**2 + 
                       (pdat['V']*pdat['Verr'])**2)/P
    stk_mask = P >= sigma * Perr
    stk_i = {}
    
    for S in "QUV":
        stk_i[S] = pdat[S][stk_mask]/P[stk_mask]

    stk_o = deepcopy(stk_i)

    if plot_on_surface:
        # get angles first
        phi, theta = cart2sph(stk_i['Q'], stk_i['U'], stk_i['V'])

        # imprint on surface of sphere
        stk_i['Q'], stk_i['U'], stk_i['V'] = sph2cart(1.0, phi, theta)


    # model stokes data
    if plot_model:
        stk_m = {}
        stk_mo = {}
        for S in "QUV":
            stk_m[S] = model_curve(stk_i[S], n = n, samp = 1000)
            stk_mo[S] = model_curve(stk_o[S], n = n, samp = 1000)

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
    
    # plot model data if applicable
    if plot_model:
        fig2, ax2 = plt.subplots(1, 1, figsize = (10, 10))
        ax2.set(xlabel = cbar_label, ylabel = "Norm Stokes")

        # plot each model and data for stokes
        for i, S in enumerate("QUV"):
            ax2.plot(np.linspace(*cbar_lims, stk_o[S].size), stk_o[S], color = default_col[i+1], label = S)
            ax2.plot(np.linspace(*cbar_lims, stk_mo[S].size), stk_mo[S], color = 'k')
        ax2.legend()


    # create colorbar
    ax_c = fig.add_axes([0.2, 0.07, 0.6, 0.02])
    ax_c.get_yaxis().set_visible(False)
    ax_c.imshow(np.linspace(0,1.0, 256).reshape(1, 256)[::-1], aspect = 'auto', extent = [*cbar_lims, 0.0, 1.0],
                cmap = 'viridis')
    ax_c.set_xlabel(cbar_label)

    ##================##
    ## PLOT END GUARD ##
    ##================##
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
        return fig

    return None