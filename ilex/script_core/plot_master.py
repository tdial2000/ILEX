##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     17/03/2024                 #
# Date (updated):     17/03/2024                 #
##################################################
# make multi tile plot                           #          
#                                                #
##################################################

## imports
from ..frb import FRB
from ..data import *
from ..utils import load_param_file, dict_get, fix_ds_freq_lims
from ..plot import _PLOT, plot_PA
from ..pyfit import fit
from ..fitting import make_scatt_pulse_profile_func
import yaml
import numpy as np
import matplotlib.pyplot as plt


class _empty:
    pass


def plot_master(parfile, plot_panels = "[S;D]", model = False, modelpar = None, 
                    modelpulses = False, filename = None):

    args = _empty
    args.parfile = parfile
    args.plot_panels = plot_panels
    args.model = model
    args.modelpar = modelpar
    args.modelpulses = modelpulses
    args.filename = filename

    # get figure parameters and flags
    figpar, flags = _init_figure(args)

    # plot data
    fig = _plot(args, figpar, flags)

    return fig



def _init_figure(args):
    """
    init figure parameters to make later
    
    """
    # make flags for panels
    flags = {'P': False, 'S': False, 'M': False, 'R': False, 'D': False}
    pID = args.plot_panels
    
    # determine if user specified or prefix
    if (pID[0] == "[") and (pID[-1] == "]"):    # user specified
        panel_str = ";" + pID[1:-1]
        axids = pID[1:-1].split(';')
        panelw = []
        h = 0
        for _, axid in enumerate(axids):
            if axid == "D":
                panelw += [2]
                flags['D'] = True
                h += 2
            else:
                panelw += [1]
                flags[axid] = True
                h += 1
        
    else:
        if pID == "PA":
            panel_str = ";P;S;D"
            panelw = [1,1,2]
            for s in "PSD":
                flags[s] = True
            h = 4
            axids = ['P', 'S', 'D']

        elif pID == "TimeFit":
            panel_str = ";M;R;D"
            panelw = [1,1,2]
            for s in "MRD":
                flags[s] = True
            h = 4
            axids = ['M', 'R', 'D']
    
    # run through each axes, making sure data is there
    enum_ax = axids.copy()
    for axs in enum_ax:
        if axs == "R":
            if ((not args.model) or ("M" not in enum_ax)) and args.modelpar is not None:
                axids.remove("R")

    # save params of figure to create later 
    figpar = {'panel_str':panel_str[1:], 'panelw':panelw, 'axids':axids, 'h': h}
    
    return figpar, flags



def _make_figure(figpar):
    """
    Make figure
    """

    panel_str = figpar['panel_str']
    panelw = figpar['panelw']
    axids = figpar['axids']
    h = figpar['h']

    # create figure
    fig, AX = plt.subplot_mosaic(panel_str, figsize = (10, 2*h),
                gridspec_kw = {"height_ratios":panelw}, sharex = True)
    for axs in axids[:-1]:
        AX[axs].get_xaxis().set_visible(False)
    AX[axids[-1]].set(xlabel = "Time offset [ms]")


    return fig, AX




def _plot(args, figpar, flags):
    """
    Plot
    """

    def plot_all_pulses(ax, x, npulse, posterior):
        """ Plot each pulse """

        single_pulse = make_scatt_pulse_profile_func(1)
        for i in range(1, npulse+1):
            y = single_pulse(x, a1 = posterior[f"a{i}"], tau = posterior['tau'],
                    mu1 = posterior[f"mu{i}"], sig1 = posterior[f"sig{i}"])

            # cut pulse at 3 sigma (roughly...)
            # mask = y > 0.003*np.max(y)

            ax.plot(x, y, '--', linewidth = 1.0)
        



    # get plotting parameters


    # create FRB instance
    frb = FRB()
    frb.load_data(yaml_file = args.parfile)

    # get data
    data_list = []
    if flags['D']:  # dynspec
        data_list += ['dsI']
    if flags['M'] or flags['R']:
        data_list += ['tI']
    if flags['P']:
        data_list += ['tQ', 'tU']
    if flags['S']:
        data_list += ['tI', 'tU', 'tQ', 'tV']
    data_list = list(set(data_list))
    print(data_list)

    data = frb.get_data(data_list, get = True)

    pars = load_param_file(args.parfile)

    if (args.model or flags['M']) and args.modelpar is None:
        # run model
        p = frb.fit_tscatt(method = pars['fits']['fitmethod'], npulse = pars['fits']['tscatt']['npulse'],
            priors = pars['fits']['tscatt']['priors'], statics = pars['fits']['tscatt']['statics'],
            fit_params = pars['fits']['tscatt']['fit_params'], redo = pars['fits']['redo'], 
            filename = args.filename) 
        NPULSES = pars['fits']['tscatt']['npulse']
    
    elif args.modelpar is not None:
        # create model
        with open(args.modelpar, "r") as file:
            model_par = yaml.safe_load(file)
        p = fit(x = data['time'], y = data['tI'], yerr = data['tIerr']*np.ones(data['tI'].size),
                    func = make_scatt_pulse_profile_func(model_par['npulse']))
        for key in model_par['posterior'].keys():
            p.set_posterior(key, model_par['posterior'][key], 0.0, 0.0)
        p._is_fit = True
        NPULSES = model_par['npulse']

        
    # create figure, make sire that figure is created after bayesian modelling, since plots are made inbetween then
    fig, AX = _make_figure(figpar)


    # plot dynamic spectra
    if flags['D']:
        ds_freq_lims = fix_ds_freq_lims(frb.this_par.f_lim, frb.this_par.df)
        AX['D'].imshow(data['dsI'], aspect = 'auto', extent = [*frb.this_par.t_lim, 
                                              *ds_freq_lims])
        AX['D'].set(ylabel = "Freq [MHz]")
    

    # plot Stokes spectra
    if flags['S']:
        frb.plot_stokes(ax = AX['S'], stk_type = "t", Ldebias = pars['plots']['Ldebias'],
                        sigma = pars['plots']['sigma'], stk_ratio = pars['plots']['stk_ratio'],
                        stk2plot = pars['plots']['stk2plot'])
        AX['S'].set(ylabel = "Flux Density (arb.)")

        # check for model 
        if args.model and (not flags['M']):
            # get model and plot
            AX['S'].plot(*p.get_model(), color = 'coral', linewidth = 2)
            if args.modelpulses:
                plot_all_pulses(AX['S'], p.x, NPULSES, p.get_post_val())


    # plot model
    if flags['M']:
        
        _PLOT(ax = AX['M'], x = p.x, y = p.y, yerr = p.yerr, 
                plot_type = pars['plots']['plot_type'], color = 'k')
                
        AX['M'].plot(*p.get_model(), color = [0.9098, 0.364, 0.3961], linewidth = 1.5)
        
        if args.modelpulses:
            plot_all_pulses(AX['M'], p.x, NPULSES, p.get_post_val())

        AX['M'].set(ylabel = "Flux Density (arb.)")

    # plot residuals
    if flags['R']:
        _PLOT(ax = AX['R'], x = p.x, y = p.y - p.get_model()[1], yerr = p.yerr,
                    plot_type = pars['plots']['plot_type'], color = 'k')
        AX['R'].set(ylabel = "Flux Density (arb.)")


    # plot Polarisation Position angle (PA)
    if flags['P']:
        PA, PAerr = calc_PAdebiased(dict_get(data,["tU", "tQ", "tUerr", "tQerr", "tIerr"]), 
                        Ldebias_threshold = pars['plots']['Ldebias_threshold'])  
        plot_PA(data['time'], PA, PAerr, ax = AX['P'], flipPA = pars['plots']['flipPA'], 
                plot_type = frb.plot_type)


    # final figure adjustments
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0)


    # save file
    if args.filename is not None:
        plt.savefig(args.filename)

    return fig






