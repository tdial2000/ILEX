########################################

# This script splits a given frb into N 
# subbands, fits scattering timescales
# per frequency and estimates the
# scattering power law index

########################################

# imports 
# ilex packages
from ilex.frb import FRB 
from ilex.pyfit import fit 
from ilex.io import ilexIO
from ilex.fitting import specindex2, gaussian, scatt_pulse_profile, scatt_pulse_profile_relative

# general packages
import argparse, os
import matplotlib.pyplot as plt
import numpy as np 
from copy import deepcopy






def get_args():

    desc = """
    This script takes an ILEX config file (.yaml) as input and reads the 
    'tscatt' dictionary of arguments 
    priors: Priors for different parameters
    statics: Values to keep constant during sampling
    npulse: Number of pulses to fit
    fit_params: additional arguments for fitting routine (.Bilby.run_sampler)

    The format for these arguments in the ILEX config file are as follows
    pars = yaml.load(.yaml)

    ## fitting ##
    This script using bayesian fitting through the .Bilby package to get the scattering 
    timescales. Then simple least squares for fitting the scattering index as a function of 
    frequency.

    pars:
        fits:
            npulse:
            fitmode: 
            statics:
            priors:
            fit_params:

    ## subbanding ##
    the subbanding is dictated by the 'f_crop' metaparameter in the ILEX config file along with 
    the -N argument for the number of subbands to split the FRB into.

    alternatively, the user can specifically define a set of subbands using the multi.fcrops param in
    the ILEX config file. This param takes in a list of fcrops that define each subband
    i.e.
    multi:
        fcrops: [[800, 900], [950:1000]]
    This will perform two subband fits.

    ## outputs ##
    You can specify the output destination and file names in one of two ways:

    1. Using the 'outdir' keyword in 'fit_params' for the directory and 'label' keyword for the file names 
       (i.e. <fit_params['label']>_*.png/npy etc.)

    2. Specify -o and -f arguments for the output directory and file name prefix (i.e. <-f>_*.png/npy etc.) 

    If -o and -f are specified, they will take precidence over the fit_params keywords in the ILEX config file.
    If none of the above are specified, the script will default to creating a directory called 'scattindex' and 
    adding the prefix 'frb' to all the files, i.e. 
    ./scattindex/frb_*.
    """

    parser = argparse.ArgumentParser(description = desc, formatter_class = argparse.RawTextHelpFormatter)

    # inputs
    parser.add_argument('--parfile', help = "ILEX config file (.yaml)", type = str, required = True)
    
    # processing arguments
    parser.add_argument('-N', help = "Number of subbands to split to fit scattering timescale", type = int, 
                        default = 1)

    # additonal arguments
    parser.add_argument('-v', help = "Verbose mode, makes more diagnostic plots", action = "store_true")
    parser.add_argument('-p', help = "Show plots", action = "store_true")
    parser.add_argument('-r', help = "Redo subband fitting", action = "store_true")

    # outputs 
    parser.add_argument('-o', help = "Output directory", type = str, default = None)
    parser.add_argument('-f', help = "Filename prefix", type = str, default = None)

    # plotting parameters
    parser.add_argument('--pw', help = "Width of dynspec to plot (visual purposes only!)", type = float, default = 150.0)
    parser.add_argument('--ptN', help = "Time downsampling of dynspec to plot (visual purposes only!)", type = int, default = 10)
    parser.add_argument('--pfN', help = "Freq downsampling of dybspec to plot (visual purposes only!)", type = int, default = 4)
    
    return parser.parse_args()





def fit_scatt(args):
    """
    Fit scattering timescales for different subbands

    Returns
    -------
    subband_fits: list[ilex.pyfit.fit]
        list of ilex.pyfit.fit objects
     
    scattindex_fit: ilex.pyfit.fit
        ilex.pyfit.fit object
    """

    subband_fits = []

    # get params for tscatt fitting
    tscatt_pars = ilexIO(filepath = args.parfile).load_pars()['fits']['tscatt']
    fcrops = ilexIO(filepath = args.parfile).load_pars()['multi']['fcrops']

    # check if label has been given, else add it
    if 'fit_params' not in tscatt_pars.keys():
        tscatt_pars['fit_params'] = {}

    if 'label' not in tscatt_pars['fit_params'].keys():
        tscatt_pars['fit_params']['label'] = "frb"

    if 'outdir' not in tscatt_pars['fit_params'].keys():
        tscatt_pars['fit_params']['outdir'] = "scattindex"

    if args.f is not None:
        print(f"Setting filenames to have the prefix <{args.f}>...")
        tscatt_pars['fit_params']['label'] = args.f 
    
    if args.o is not None:
        print(f"Setting output directory to <{args.o}>...")
        tscatt_pars['fit_params']['outdir'] = args.o

    # add pars to args for later use
    if 'npulse' in tscatt_pars.keys():
        args.npulse = tscatt_pars['npulse']
    else:
        args.npulse = 1
    args.o = tscatt_pars['fit_params']['outdir']
    args.f = tscatt_pars['fit_params']['label']

    if 'fitmode' in tscatt_pars.keys():
        args.fitmode = tscatt_pars['fitmode']
    else:
        args.fitmode = "abs"

    if args.fitmode == "abs":
        args.scattfunc = scatt_pulse_profile
    elif args.fitmode == "relative":
        args.scattfunc = scatt_pulse_profile_relative
    else:
        ValueError(f"fitmode = {fitmode} is not supported!")

    args.tscatt_pars = deepcopy(tscatt_pars)

    # load data and get freq array
    frb = FRB(args.parfile)
    frb.set(verbose = args.v, show_plots = False, save_plots = False)

    bw = frb.metapar.f_crop[1] - frb.metapar.f_crop[0]                          # bw
    freq_edges = np.linspace(*frb.metapar.f_crop, args.N+1)                      # freq edges
    freq_bins = []
    freq_c = []
    for i in range(args.N):
        freq_bins += [[freq_edges[i], freq_edges[i+1]]]
        freq_c += [(freq_edges[i] + freq_edges[i+1])/2]

    if fcrops is not None:
        freq_bins = fcrops.copy()
        freq_c = []
        args.N = len(fcrops)
        for fcrop in fcrops:
            freq_c += [(fcrop[1] + fcrop[0])/2]
    freq_c = np.array(freq_c)

    # add freq bins to args for later use
    args.freq_c = freq_c
    args.freq_bins = freq_bins
    args.fmin = frb.metapar.f_crop[0]
    args.fmax = frb.metapar.f_crop[1]
    
    # create empty arrays for tscatt
    tscatt = np.zeros(args.N)
    tscatt_err = np.zeros(args.N)
    
    print("## Performing subband fit for scattering index ##")
    print("-"*50)

    print(f"Number of subbands: {args.N}")
    print(f"Bandwidth: {bw}")
    print(f"Bandwidth per subband: {bw/args.N}")
    print(f"Subband central frequencies: {freq_c}")

    args.tcrop = frb.metapar.t_crop.copy()


    for i in range(args.N):
        print(f"Fitting subband {i}: freq = {freq_bins[i]}")
        tscatt_pars['fit_params']['label'] = args.f + f"_subband{i}"
        subband_fits += [frb.fit_tscatt(method = "bayesian", f_crop = freq_bins[i],
                            **tscatt_pars, redo = args.r)]
        
        tscatt[i] = subband_fits[i].get_post_val()['tau']
        tscatt_err[i] = subband_fits[i].get_mean_err()['tau']

    # get scattering values and fit scatt index
    scattindex_fit = fit(method = "least squares", x = freq_c, y = tscatt, yerr = tscatt_err,
                         func = specindex2, fit_keywords = {'maxfev':200000}, residuals = False)
    
    scattindex_fit.fit()
    # scattindex_fit.stats()

    # print out final values
    print("Final fitted values:\n")
    coltitlestr = "band".ljust(5) + " |" + "cfreq".ljust(10) + " |" + "bw".ljust(10) + " |"
    keys = subband_fits[i].get_post_val().keys()
    for key in keys:
        static_str = ""
        if subband_fits[i].static[key] is not None:
            static_str = " (static)"
        coltitlestr += (f"{key}" + static_str).ljust(20) + " |"
    print(coltitlestr)
    print("-"*(22*len(keys) + 31))
    for i in range(args.N):
        vals, errs = subband_fits[i].get_post_val(), subband_fits[i].get_mean_err()
        subband_str = f"{i}".ljust(5) + " |" + f"{args.freq_c[i]:.4f}".ljust(10)
        subband_str += " |" + f"{args.freq_bins[i][1] - args.freq_bins[i][0]:.4f}".ljust(10) + " |"
        for j, key in enumerate(keys):
            subband_str += f"{vals[key]:.4f} +/- {errs[key]:.4f}".ljust(20) + " |"
        print(subband_str)
    print("-"*(22*len(keys) + 31))
    print("\n")



    return subband_fits, scattindex_fit








def plot_scatt(args, subband_fits, scattindex_fit):
    """
    Plots various "stuff" concerning scattering
    
    """


    ## SUBBAND PLOT ##
    # make colorbar
    cmap = plt.colormaps['jet']
    col = cmap(np.linspace(0, 1, args.N))
    if args.v:
        fig, ax = plt.subplots(1, 2, figsize = (12,12), gridspec_kw = {'width_ratios':[24, 1]})
        ax = ax.flatten()

        # get incremental y scale to add to each line plot
        amp = 0
        for i in range(args.N):
            arr_amps = []
            for j in range(args.npulse):
                arr_amps += [subband_fits[i].get_post_val()[f"a{j+1}"]]
            amp += max(arr_amps)
        amp /= args.N 
        yoffset = 0.2 * amp # arbitrary, used to offset all subbands in plot

        # plot subbands
        for i in range(args.N):
            xi = np.linspace(subband_fits[i].x[0], subband_fits[i].x[-1], 1000)
            ax[0].errorbar(subband_fits[i].x, subband_fits[i].y + yoffset*i, subband_fits[i].yerr, 
                            markeredgecolor = col[i], alpha = 0.6, marker = 'o', color = col[i], capsize = 2.0, 
                            linestyle = "")
            ax[0].plot(xi, subband_fits[i].get_model(x = xi)[1] + yoffset*i, color = col[i], linewidth = 1.0)
            
        ax[0].set_xlabel("Time [ms]", fontsize = 16)
        ax[0].get_yaxis().set_visible(False)
        
        # plot colorbar
        ax[1].imshow(args.freq_c.reshape(args.N, 1)[::-1],
                    aspect = 'auto', extent = [0, 1, args.fmin, args.fmax], cmap = 'jet')
        ax[1].get_xaxis().set_visible(False)
        ax[1].set_ylabel("Frequency [MHz]", fontsize = 16)
        ax[1].set_yticks(args.freq_c, args.freq_c)
        ax[1].get_yaxis().set_label_position('right')
        ax[1].get_yaxis().tick_right()

        fig.tight_layout()
        fig.subplots_adjust(wspace = 0)

        plt.savefig(os.path.join(args.o, args.f + "_subbands.png"))



    ## SCATT INDEX PLOT
    # scattindex_fit.plot(xlabel = "Frequency [MHz]", ylabel = "$\\tau$ [ms]", show = False)
    # plt.gca().set_title(f"$\\alpha$ = {scattindex_fit.get_post_val()['alpha']:.4f} +/- {scattindex_fit.get_mean_err()['alpha']:.4f}")
    # plt.gcf().tight_layout()

    fig2, ax2 = plt.subplots(1, 1, figsize = (10, 8))

    # plot fitted scatt index
    scattindex_fit.set_plot_vars(plot_model_kwargs = { 
        'label':f"[bestfit] $\\alpha$ = {scattindex_fit.get_post_val()['alpha']:.4f} +/- {scattindex_fit.get_mean_err()['alpha']:.4f}"})
    scattindex_fit.plot_model_on_ax(ax = ax2)
    
    # make plot with alpha = -4
    scattindex_fittemp = deepcopy(scattindex_fit)
    scattindex_fittemp.set(static = {'alpha':-4.0})
    scattindex_fittemp.fit(redo = True)

    scattindex_fittemp.set_plot_vars(plot_model_kwargs = {'label':f"$\\alpha$ = -4.0", 'color':'dodgerblue'},
                                 plot_err_kwargs = {'capsize':4.0})
    scattindex_fittemp.plot_model_on_ax(ax = ax2)

    # set axes properties
    ax2.set_yscale('log')
    ax2.set_xlabel("Frequency [MHz]", fontsize = 16)
    ax2.set_ylabel("$\\tau$ [ms]", fontsize = 16)
    ax2.legend()

    fig2.tight_layout()

    plt.savefig(os.path.join(args.o, args.f + "_scattindex.png"))


    ## plot dynamic spectrum and subbands 
    # first subplot is full frequency band, second plot is zoomed in f_crop
    frb = FRB(args.parfile)

    tcrop = frb.metapar.t_crop.copy()
    remw = args.pw - (tcrop[1] - tcrop[0])
    tcrop[0] -= remw/2
    tcrop[1] += remw/2

    # check widths
    if tcrop[0] < frb.par.t_lim[0]:
        tcrop[0] = frb.par.t_lim[0]
    if tcrop[1] > frb.par.t_lim[1]:
        tcrop[1] = frb.par.t_lim[1]

    fig3, ax3 = plt.subplots(1, 2, figsize = (14,7), sharex = True)
    ax3 = ax3.flatten()
    frb.plot_data_on_axes("dsI", ax = ax3[0], f_crop = ['min', 'max'], show_plots = False, t_crop = tcrop,
                    tN = args.ptN, fN = args.pfN)
    frb.plot_data_on_axes("dsI", ax = ax3[1], show_plots = False, t_crop = tcrop, tN = args.ptN,
                    fN = args.pfN)
    ax3[1].get_yaxis().set_visible(False)
    ax3[1].set_ylim(frb.this_par.f_lim)

    # plot subbands on each 
    xlim = ax3[0].get_xlim()
    xwidth = xlim[-1] - xlim[0]
    for i in range(args.N):
        for j in range(2):
            # plot patches on LHS
            ax3[j].fill_between([xlim[0], xlim[0] + 0.025 * xwidth], *args.freq_bins[i], 
                                color = col[i], label = f"f$_{{c}}$ = {args.freq_c[i]:.2f} MHz")
    
    ax3[0].legend()
    fig3.tight_layout()
    fig3.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(os.path.join(args.o, args.f + "_dynspec.png"))

    ## plot each subband and all the gaussians that make that subband up
    fig4, ax4 = plt.subplots(args.N, 1, figsize = (8, 3 * args.N), sharex = True)
    if args.N > 1:
        ax4 = ax4.flatten()[::-1]

    # make colorbar
    cmap = plt.colormaps['jet']
    col = cmap(np.linspace(0, 1, args.N))

    # plot subbands
    npulse = 0
    i = 0
    while True:
        i += 1
        if not f"mu{i}" in subband_fits[0].get_post_val().keys():
            break

        npulse += 1
    for i in range(args.N):
        xi = np.linspace(subband_fits[i].x[0], subband_fits[i].x[-1], 1000)
        ax4[i].errorbar(subband_fits[i].x, subband_fits[i].y, subband_fits[i].yerr, 
                        markeredgecolor = col[i], alpha = 0.6, marker = 'o', color = col[i], capsize = 2.0, 
                        linestyle = "")
        ax4[i].plot(xi, subband_fits[i].get_model(x = xi)[1], color = col[i], linewidth = 1.5)
        vals = subband_fits[i].get_post_val()
        for j in range(1, npulse + 1):
            pos = vals[f'mu{j}']
            if (j > 1) and (args.fitmode == "relative"):
                pos = vals[f'mu{j}'] + vals['mu1'] 
            ax4[i].plot(xi, scatt_pulse_profile(xi, a1 = vals[f"a{j}"], mu1 = pos, 
                                sig1 = vals[f"sig{j}"], tau = vals['tau']), color = 'k', linewidth = 1.0,
                                linestyle = '--')
        
        if i == args.N - 1:
            ax4[i].set_ylabel("Flux Density (arb.)", fontsize = 16)
        ax4[i].set_xlabel("Time [ms]", fontsize = 16)

    fig4.tight_layout()
    fig4.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(os.path.join(args.o, args.f + "_subbands_seperated.png"))
    

    if args.p:
        plt.show()




def save_data(args, subband_fits, scattindex_fits):
    with open(os.path.join(args.o, args.f + "_summary.txt"), 'w') as file:
        file.write("### Time in [ms] and Frequency in [MHz] ###\n")
        alpha_post = scattindex_fits.get_posteriors()
        file.write(f"alpha: {alpha_post['alpha'].val} +{alpha_post['alpha'].p}/-{alpha_post['alpha'].m}\n")
        file.write(f"tau_c: {alpha_post['tau_c'].val} +{alpha_post['tau_c'].p}/-{alpha_post['tau_c'].m} (see ILEX documentation for method of fitting scattering index)\n")
        file.write(f"# subbands: {args.N}\n")
        file.write(f"tcrop: {args.tcrop}\n")
        file.write(f"Priors: {args.tscatt_pars['priors']}\n")
        file.write(f"statics: {args.tscatt_pars['statics']}\n")
        file.write(f"# pulses: {args.npulse}\n")
        file.write(f"taus: ")
        taus, taus_n, taus_p = [], [], []
        for i in range(args.N):
            posteriors = subband_fits[i].get_posteriors()
            taus += [posteriors['tau'].val]
            taus_n += [posteriors['tau'].m]
            taus_p += [posteriors['tau'].p]
        file.write(str(taus))
        file.write("\n")
        file.write("taus +err: ")
        file.write(str(taus_p))
        file.write("\n")
        file.write("taus -err: ")
        file.write(str(taus_n))
        file.write("\n")
        file.write("freqs: ")
        file.write(str(list(args.freq_c)))
        file.write("\n")
        file.write("bw: ")
        bw = []
        for i in range(args.N):
            bw += [args.freq_bins[i][1] - args.freq_bins[i][0]]
        file.write(str(bw))
        file.write("\n")



if __name__ == "__main__":
    # main block of code
    
    args = get_args()


    # fit scattering index
    subband_fits, scattindex_fit = fit_scatt(args)


    # plotting
    plot_scatt(args, subband_fits, scattindex_fit)

    # save data
    save_data(args, subband_fits, scattindex_fit)


    print("[fit_scattindex.py]: End of script!!")


