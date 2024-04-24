##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     17/03/2024                 #
# Date (updated):     17/03/2024                 #
##################################################
# Plot multiple PA components                    #          
#                                                #
##################################################

## imports
from ..frb import FRB
from ..data import *
from ..utils import load_param_file, dict_get
from ..plot import _PLOT, plot_PA, plot_stokes
import yaml
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor


class _empty:
    pass


def plot_PA_multi(parfile, RMplots = False, RMburst = False, showbounds = False, 
                    filename = None):
    
    args = _empty
    args.parfile = parfile
    args.RMplots = RMplots
    args.RMburst = RMburst
    args.showbounds = showbounds
    args.filename = filename

    fig = _PA_multi(args)

    return fig




def _PA_multi(args):
    """
    Plot multiple PA plots

    """

    # create figure
    fig_PA, AX_PA = plt.subplot_mosaic("P;S;D", figsize = (12, 10), 
            gridspec_kw={"height_ratios": [1, 2, 2]}, sharex=True)



    # create FRB object
    frb = FRB()
    frb.load_data(yaml_file = args.parfile)



    # get pars from parfile
    pars = load_param_file(args.parfile)
    PLOT_TYPE = pars['plots']['plot_type']



    # get tcrops 
    tcrops = pars['multi']['tcrops']
    fcrops = pars['multi']['fcrops']
    
    if tcrops is None and fcrops is None:
        print("Must specify tcrops and fcrops in par.yaml file!")
        sys.exit()     


    ncomps = len(tcrops)
    if tcrops is None:
        tcrops = [frb.metapar.t_crop] * len(fcrops)
    if fcrops is None:
        fcrops = [frb.metapar.f_crop] * len(tcrops)

    



    # loop over tcrops, fitting for RM
    RMcomp_fits = []
    for i, tcrop in enumerate(tcrops):
        # fit for RM
        p = frb.fit_RM(t_crop = tcrop, f_crop = fcrops[i], method = pars['fits']['fitRM']['method'], 
            fit_params = pars['fits']['fitRM']['fit_params'], plot = False)
        RMcomp_fits += [p]

        # reset tcrops and fcrops
        tcrops[i] = frb.prev_metapar.t_crop.copy()
        fcrops[i] = frb.prev_metapar.f_crop.copy()

    t_crophase, _ = frb.par.lim2phase(t_lim = frb.metapar.t_crop)

    # get full tcrop bounds
    tcrop_full = [min(np.min(tcrops), t_crophase[0]), max(np.max(tcrops), t_crophase[1])]
    fcrop_full = [np.min(fcrops), np.max(fcrops)]


    # get full data to use for plotting
    data_full = frb.get_data(["tI", "tQ", "tU", "tV", "dsI"], t_crop = tcrop_full,
                    f_crop = fcrop_full, get = True)    
    dt = round(data_full['time'][1] - data_full['time'][0], 6)



    # make nan array for PA
    PA = np.full(data_full['tQ'].size, np.nan)
    PAerr = PA.copy()



    # loop over tcrops again, getting de-rotated stokes data and to calculate
    # debiased PA
    # Get de-rotated stokes and plot
    comp_times = []
    for i, tcrop in enumerate(tcrops):
        # get derotated data
        vals = RMcomp_fits[i].get_post_val(func = False)
        data_i = frb.get_data(["tI", "tQ", "tU"], t_crop = tcrop, f_crop = fcrops[i],
                RM = vals['rm'], f0 = vals['f0'], get = True)

        comp_times += [[*data_i['time'][[0, -1]]]]

        # overlay de-rotated stokes data to full stokes spectra
        xs = int(round((data_i['time'][0] - data_full['time'][0])/dt))
        ns = data_i['time'].size

        for S in "QU":
            data_full[f"t{S}"][xs:xs+ns] = data_i[f"t{S}"]

        # calculate debiased PA then overlap on full PA array
        PA_i, PAerr_i = calc_PAdebiased(data_i, Ldebias_threshold = pars['plots']['Ldebias_threshold'])
        PA[xs:xs + ns], PAerr[xs:xs + ns] = PA_i, PAerr_i
        


    # plot
    # plot PA
    plot_PA(data_full['time'], PA, PAerr, ax = AX_PA['P'], flipPA = pars['plots']['flipPA'],
        plot_type = "scatter")



    # plot stokes
    plot_stokes(data_full, stk_type = "t", ax = AX_PA['S'], Ldebias = pars['plots']['Ldebias'],
            sigma = pars['plots']['sigma'], plot_type = PLOT_TYPE,
            plot_L = pars['plots']['plot_L'], stk_ratio = pars['plots']['stk_ratio'],
            stk2plot = pars['plots']['stk2plot'])

    # plot bounds
    if args.showbounds:
        ylim = AX_PA['S'].get_ylim()
        for _, crop in enumerate(tcrops):
            crop,_ = frb.par.phase2lim(t_crop = crop)
            AX_PA['S'].fill_between(crop, ylim[0], ylim[1], color = [0.3, 0.3, 0.3], alpha = 0.3)
        AX_PA['S'].set_ylim(ylim)



    # plot stokes Dynamic spectrum I
    AX_PA['D'].imshow(data_full['dsI'], aspect = 'auto', extent = [*data_full['time'][[0,-1]], 
            np.min(data_full['freq']), np.max(data_full['freq'])])
    AX_PA['D'].set(xlabel = "Time offset [ms]", ylabel = "Freq [MHz]")



    # final figure adjustments
    fig_PA.tight_layout()
    fig_PA.subplots_adjust(hspace = 0)
    AX_PA['P'].get_xaxis().set_visible(False)
    AX_PA['S'].get_xaxis().set_visible(False)



    # save figure to file
    if args.filename is not None:
        plt.savefig(args.filename + "PA_plot")



    # plot grid of RM fits for each component
    if args.RMplots:

        nrows = int(ceil(ncomps/3))
        fig_RM, AX_RM = plt.subplots(nrows, 3, figsize = (10,3*nrows))

        botax = range(ncomps - 1 - (ncomps - 1)%3, ncomps)
        AX_RM = AX_RM.flatten()
        for i in range(ncomps):
            _PLOT(ax = AX_RM[i], x = RMcomp_fits[i].x, y = RMcomp_fits[i].y, 
                yerr = RMcomp_fits[i].yerr, plot_type = frb.plot_type, color = 'k')
            AX_RM[i].plot(RMcomp_fits[i].x, RMcomp_fits[i].get_model()[1], 'r', linewidth = 2)
            if i in botax:
                AX_RM[i].set_xlabel("Freq [MHz]")
            if i % 3 == 0:
                AX_RM[i].set_ylabel("PA [deg]")
            AX_RM[i].set(title = f"Component {i+1}")

            if pars['plots']['flipPA']:
                AX_RM[i].set_ylim([0, 180])
            else:
                AX_RM[i].set_ylim([-90, 90])
        
        # turn off extra axes
        for i in range(ncomps, nrows*3):
            AX_RM[i].set_axis_off()

        # save figure to file
        if args.filename is not None:
            plt.savefig(args.filename + "RM_fits")



    # plot RM variability 
    if args.RMburst:
        fig_B, ax_B = plt.subplots(1, 1, figsize = (8,6))
        # plot Stokes I time burst
        _PLOT(ax = ax_B, x = data_full['time'], y = data_full['tI'], 
            yerr = data_full['tIerr'], plot_type = frb.plot_type, color = 'k')

        ax_B.set(ylabel = "Flux Density (arb.)")

        # plot RM points
        _rm = np.zeros(ncomps)
        _rmerr = _rm.copy()
        _rmx = _rm.copy()
        for i, rmcomp in enumerate(RMcomp_fits):
            _rm[i] = rmcomp.posterior['rm'].val
            _rmerr[i] = rmcomp.get_mean_err()['rm']
            _rmx[i] = comp_times[i][0] + (comp_times[i][-1] - comp_times[i][0])/2

        # make twin axes
        ax_B2 = ax_B.twinx()
        
        ax_B2.scatter(_rmx, _rm, c = 'b', marker = "v", s = 5)
        ax_B2.errorbar(_rmx, _rm, _rmerr, color = 'b', alpha = 0.5, linestyle = '',
            capsize = 3)

        ax_B2.set_ylabel("RM $[rad/m^{{2}}]$", color = 'tab:blue')
        ax_B2.tick_params(axis='y', labelcolor='tab:blue')

        # save figure to file
        if args.filename is not None:
            plt.savefig(args.filename + "RM_burst")


    return fig_PA