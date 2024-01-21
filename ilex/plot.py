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

## import stats functions ##
from .fitting import (lorentz, scatt_pulse_profile, scat,
                       gaussian)

from .data import *
# constants
c = 2.997924538e8 # Speed of light [m/s]
default_col = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_data(data, typ = "ds", tlim = None, flim = None, ax = None, filename = None):
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
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])



    if tlim is None:
        tname = "Time (phase)"
        tlim = [0.0, 1.0]
    else:
        tname = "Time [ms]"
    
    if flim is None:
        fname = "Freq (phase)"
        flim = [0.0, 1.0]
    else:
        fname = "Freq [MHz]"
    


    # check type 
    ndims = data.ndim
    if ndims == 2 and typ == "ds":
        # plot dynspec
        ax.imshow(data, aspect = 'auto', extent = [*tlim, *flim])
        ax.set(xlabel = tname, ylabel = fname)
    
    elif ndims == 2:
        if typ == "t":
            # scrunch in freq
            tdat = np.mean(data, axis = 0)
            tx = np.linspace(*tlim, tdat.size)
            ax.plot(tx, tdat, 'k')
            ax.set(xlabel = tname, ylabel = "Flux Density (arb.)")

        elif typ == "f":
            # scrunch in time
            fdat = np.mean(data, axis = 1)
            fx = np.linspace(*flim, fdat.size)
            ax.plot(fx, fdat, 'k')
            ax.set(xlabel = fname, ylabel = "Flux Density (arb.)")

        else:
            print("Invalid data type to plot")
    
    elif ndims == 1:
        if typ == "t":
            # plot time series
            tx = np.linspace(*tlim, data.size)
            ax.plot(tx, data, 'k')
            ax.set(xlabel = tname, ylabel = "Flux Density (arb.)")
        
        elif typ == "f":
            # plot spectra
            fx = np.linspace(*flim, data.size)
            ax.plot(fx, data, 'k')
            ax.set(xlabel = fname, ylabel = "Flux Density (arb.)")
        
        else:
            print("Invalid data type to plot")
        
    else:
        print("Data is invalid")

    

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
def plot_scintband(x, y, y_err = None, w = 0.0, a = 0.0, ax = None, filename: str = None):
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


    print(ax)

    
    # start setting up axes
    ax.set_xlabel("Frequency Lag [Mhz]", fontsize = 12)
    ax.set_ylabel("normalised acf", fontsize = 12)

    # plot data
    ax.scatter(x, y, c = 'k', s = 1.5)

    # plot error bars if applicable
    if y_err is not None:
        # error bars
        ax.errorbar(x, y, yerr = y_err, fmt = 'none', ecolor = [0., 0., 0., 0.5])

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
def plot_tscatt(x, y, y_err = None, ax = None, p = None, npulse: int = 1, filename: str = None):
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
        ax.errorbar(x, y, yerr = y_err, fmt = 'none', ecolor = [0., 0., 0., 0.5])


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










def plot_RM(Q, U, Qerr = None, Uerr = None, f = None, rm = 0.0, pa0 = 0.0, f0 = 0.0,
            ax = None, filename: str = None):
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





    # set up axes
    ax.set_xlabel("Frequency [MHz]", fontsize = 12)
    ax.set_ylabel("PA [deg]", fontsize = 12)

    # calc PA
    PA = 0.5 * np.arctan2(U, Q)
    PA = np.unwrap(PA, period = np.pi)
    PA_fit = (pa0 + rm*c**2/1e12*(1/f**2 - 1/f0**2))

    # plot 
    ax.scatter(f, PA * 180/np.pi, c = 'k', s = 5, label  = "Measured PA")                # PA from data
    ax.plot(f, PA_fit * 180/np.pi, 'r', label = f"RM: {rm:.3f},    pa0: {pa0:.3f}")      # PA best fit line
    

    # error plotting
    if Qerr is not None and Uerr is not None:
        _, PA_err = calc_PA(Q, U, Qerr, Uerr)
        ax.errorbar(f, PA * 180/np.pi, PA_err * 180/np.pi, fmt = 'none', ecolor = [0., 0., 0., 0.5])
    
    # ax.set_ylim([-90, 90])

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













def plot_PA(x, PA, PA_err, ax = None, filename: str = None):
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
    
    # plot PA
    PA_mask = ~np.isnan(PA)
    ax.scatter(x[PA_mask], PA[PA_mask] * 180/np.pi, c = 'k', s = 2)
    ax.errorbar(x[PA_mask], PA[PA_mask] * 180/np.pi, PA_err[PA_mask] * 180/np.pi,
                 ecolor= 'k', linestyle = "")

    # ax.set_xlim([t_x[0], t_x[-1]])
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














def plot_stokes(x, stk_data, stk_err = None, L = False, ax = None, stk_type = "f", filename: str = None):
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

    # chek if stokes errors are given
    err_flag = False
    if stk_err is not None:
        err_flag = True
        # check if IQUV here
        for S in "IQUV":
            if S not in stk_err.keys():
                print(f"Missing stokes error {S}")
                err_flag = False


    ## check if frequency or time data
    if stk_type == "t":
        ax.set_xlabel("Time [ms]", fontsize = 12)
    elif stk_type == "f":
        ax.set_xlabel("Frequency [MHz]", fontsize = 12)
    else:
        ax.set_xlabel("arb.", fontsize = 12)

    ## update ax
    ax.set_ylabel("Flux Density. ", fontsize = 12)

    ## plot
    stks = "IQUV"
    col = default_col
    if L:
        stk_data['L'] = calc_L(stk_data['Q'], stk_data['U'])
        if err_flag:
            stk_err['L'] = np.sqrt(stk_data['Q']**2*stk_err['Q']**2 + stk_data['U']**2*stk_err['U']**2)/stk_data['L']
        stks = "ILV"
        col = ['k', 'r', 'b']
    
    for i, S in enumerate(stks):
        ax.plot(x, stk_data[S], color = col[i], label = S)
        if err_flag:
            # ax.plot(t_x, stk_err[S], color = col[i], linestyle = '--')
            ax.errorbar(x, stk_data[S], stk_err[S], ecolor= col[i], linestyle = "")
    # ax.set_xlim([np.min(t_x), np.max(t_x)])
    
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





