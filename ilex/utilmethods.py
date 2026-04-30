##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 15/12/2025
##
## Meta methods that use .FRB class and don't nessesarily
## Need to be in a script.
## 
## Util methods 
##===============================================##
##===============================================##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ilex.data import average, get_zapstr, gaussian_smooth
from ilex.widths import find_optimal_fluence_width
from ilex.plot import plot_dynspec

def findfrb_fluence(frb, yfrac = 0.95, stDev = 0, itcrop = ['min', 'max'],
                        ifcrop = ['min', 'max'], mode = "min", _iter=0):
    """
    Find Time and frequency bounds of Burst using fluence threshold

    Parameters
    ----------
    frb : ilex.FRB
        FRB instance
    yfrac : float
        fluence thrshold [0.0, 1.0], by default 0.95
    stDev: int
        HWFM (in samples) of Gaussian smoothing kernel to apply in the time domain. If zero Gaussian smoothing is skipped, by default 0
    ifcrop: list
        initial frequency crop to use. Can be used incase the burst is narrow and/or low SNR
    mode: str
        Mode for fluence cropping, by default 'min'
    iter: int
        Iteration of findfrb_fluence, only for figure naming

    Returns
    -------
    tcrop : list
        Time crop
    fcrop : list
        Freq crop
    
    """
    plot_flag = False
    if frb.save_plots:
        plot_flag = True

    # make figure
    if plot_flag:
        fig, ax = plt.subplots(2,2, figsize = (10, 10), gridspec_kw = {'width_ratios': [5, 1], 'height_ratios': [1, 5]})
        ax = ax.flatten()
        ax[1].set_axis_off()

    # find time crop
    tI = frb.get_data("tI", t_crop = itcrop, f_crop = ifcrop, get = True)['tI']
    centroid, lw, rw = find_optimal_fluence_width(tI = tI, yfrac = yfrac, mode = mode)
    tcrop = [itcrop[0]+(centroid - lw) * frb.this_par.dt, itcrop[0]+(centroid + rw) * frb.this_par.dt]


    # plot searching
    if plot_flag:
        plot_fluence_search(I = tI, centroid = centroid, lw = lw, rw = rw, _type = 't', ax = ax[0])

    tW = True
    print(f"WEIGHTS: {frb.par.tW.get_weights()}")
    if frb.par.tW.get_weights() is None:
        tW = False
    
    resetweights = False
    if (stDev > 0) and (~tW):

        # get nsamps
        tI_size = frb.get_data("tI", tN = 1, t_crop = tcrop, get = True)['tI'].size

        resetweights = True
        tIsmth = frb.get_data("tI", f_crop = ifcrop, t_crop = tcrop, get = True)['tI']
        gaussian_W = gaussian_smooth(tIsmth, stDev)
        tIsmth = np.interp(np.linspace(0, 1.0, tI_size),
                np.linspace(0, 1.0, gaussian_W.size), gaussian_W)
        wavg = np.sum(tIsmth)
        tW = True
        frb.par.set_weights(xtype = "t", method = "None", W = tIsmth)

    # find freq crop
    fI = frb.get_data("fI", t_crop = tcrop, f_crop = ['min', 'max'], get = True)['fI']
    fIzeroed = fI.copy()
    fIzeroed[np.isnan(fIzeroed)] = 0.0
    centroid_f, lw_f, rw_f = find_optimal_fluence_width(tI = fIzeroed, yfrac = yfrac, mode = "min")
    fcrop = [frb.this_par.f_lim[1] - (centroid_f + rw_f) * frb.this_par.df, 
                frb.this_par.f_lim[1] - (centroid_f - lw_f) * frb.this_par.df]


    if plot_flag:
        plot_fluence_search(I = fI, centroid = centroid_f, lw = lw_f, rw = rw_f, _type = 'f', ax = ax[3])

    print(f"Crops found: t: {tcrop}, f: {fcrop} -> iter: {_iter+1}")

    # reset weights
    if resetweights:
        frb.par.set_weights(xtype = 't', method = "None", W = None)

    if plot_flag:
        dsI = frb.get_data('dsI', t_crop = itcrop, f_crop = ifcrop, 
                            tN = frb.this_metapar.tN, fN = frb.this_metapar.fN, get = True)['dsI']
        plot_dynspec(dsI, aspect = 'auto', ax = ax[2])
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0)
        fig.savefig(f"{frb.this_par.name}_findfrb{_iter+1}.png")
        plt.close(fig)

    return tcrop, fcrop, centroid, lw, rw


def plot_fluence_search(I, centroid, lw, rw, _type = 't', ax = None):
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize = (8, 6))
        new_fig = True
    
    if _type == 't':
        x = np.arange(I.size)
        ax.plot(x, I, color = 'k')
        ylim = ax.get_ylim()
        ax.plot([centroid - lw]*2, ylim, '--r')
        ax.plot([centroid + rw]*2, ylim, '--r')
        xlim = ax.get_xlim()
        ax.plot(xlim, [0,0], color = [0.4,0.4,0.4], zorder = 0)
        ax.set_xlim([x[0], x[-1]])
    else:
        y = np.arange(I.size)
        ax.plot(I[::-1], np.arange(I.size), color = 'k')
        xlim = ax.get_xlim()
        ax.plot(xlim, [I.size - centroid + lw]*2, '--r')
        ax.plot(xlim, [I.size - centroid - rw]*2, '--r')
        ylim = ax.get_ylim()
        ax.plot([0,0], ylim, color = [0.4,0.4,0.4], zorder = 0)
        ax.set_ylim([y[0], y[-1]])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if new_fig:
        fig.tight_layout()
        return fig
    return 

