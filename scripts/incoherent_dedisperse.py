##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     18/11/2024                 #
# Date (updated):     18/11/2024                 #
##################################################
# Incoherently maximise DM of Stokes I dynspec   #          
#                                                #
##################################################

# imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ilex.data import average
from scipy.optimize import curve_fit


def get_args():
    """
    Get arguments
    """

    parser = argparse.ArgumentParser()

    # data input
    parser.add_argument("-i", help = "Stokes I dynamic spectrum (.npy): reference frequency assumed bottom of the band", type = str)
    
    # pre-processing arguments
    parser.add_argument("--dt", help = "Time resolution in ms", type = float, default = 0.001)
    parser.add_argument("--tN", help = "Time averaging factor", type = int, default = 1)
    
    # DM arguments
    parser.add_argument("--DMmin", help = "Minimum of DM [pc/cm^3] range to search over", type = float, default = -1.0)
    parser.add_argument("--DMmax", help = "Maximum of DM [pc/cm^3] range to search over", type = float, default = 1.0)
    parser.add_argument("--DMstep", help = "Step size of DM [pc/cm^3]", type = float, default = 0.1)

    # bandwidth parameters
    parser.add_argument("--cfreq", help = "Central frequency [MHz] of Dynamic spectrum", type = float, default = 919.5)
    parser.add_argument("--bw", help = "Bandwidth [MHz] of Dynamic spectrum", type = float, default = 336)
    parser.add_argument("--lower", help = "Use if first channel is bottom of the band", action = "store_true")

    # additional arguments
    parser.add_argument("-o", help = "Output filename, No output saved if not specified", type = str, default = None)
    parser.add_argument("--delDM", help = "Delta DM [pc/cm^3] to apply for dedispersion, if given will overide search", type = float, default = None)
    parser.add_argument("--quadfit", help = "Fit sn vs dm to a quadratic and extract optimal DM", action = "store_true")

    args = parser.parse_args()

    # add constants
    args.kdm = 4149.377593

    return args





def search_DM(args):
    """
    Incoherently search through DM range 

    """

    print("Performing De-dispersion")
    print("="*25, "\n")
    print(f"dt: {args.dt} [ms] -> {args.tN} * {args.dt} = {args.tN * args.dt} [ms]")
    print("cfreq:".ljust(15) +  f"{args.cfreq} [MHz]")
    print("bw:".ljust(15) + f"{args.bw} [MHz]\n")
    print("Min DM  |  Max DM  |  DM step    [pc/cm^3]")
    print("-"*29)
    print(f"{args.DMmin:.4f}".ljust(8) + "| " + f"{args.DMmax:.4f}".ljust(9)  + "| " + f"{args.DMstep:.4f}\n")


    # load in data and create freq array
    with open(args.i, "rb") as file:
        dynI = np.load(file)
    
    args.nchan, args.nsamp = dynI.shape
    args.df = args.bw / args.nchan

    freq = np.linspace(args.cfreq + args.bw/2 - args.df,
                       args.cfreq - args.bw/2 + args.df, args.nchan)
    
    if args.lower:
        freq = freq[::-1]

    args.f0 = np.min(freq)
    
    # pre-process
    dynI = average(dynI, N = args.tN, axis = 1)

    # set up secondary arrays to perform DM search
    dedispersed_dynI = dynI.copy()                                 # test dynamic spectrum
    trial_dms = np.arange(args.DMmin, args.DMmax, args.DMstep)     # dm trial
    trial_dm_vals = np.zeros(trial_dms.size)                       # dm trial values
    args.ntrials = trial_dms.size

    # get flagged channels before hand to speed things up
    flagged_chans = np.isnan(dynI[:,0])

    # k val is the peak value of the burst at each dm trial
    for i, trial_dm in enumerate(trial_dms):
        print(f"Searching trial DMs...   {i/args.ntrials:.2%}", end = "\r")
        
        dm_shifts = (args.kdm * trial_dm * (1/freq**2 - 1/args.f0**2) / (args.dt * args.tN * 1e-3)).astype(int)

        for j, dm_shift in enumerate(dm_shifts):
            if not flagged_chans[j]:
                dedispersed_dynI[j] = np.roll(dynI[j], -dm_shift)
        
        # scrunch to get time series and find maximum of time series
        trial_dm_vals[i] = np.max(np.mean(dedispersed_dynI[~flagged_chans], axis = 0))
    
    print(f"Searching trial DMs...   100.00%\n")

    def quadratic(x, a, b, c):
        return a*x**2 + b*x + c

    if args.quadfit:
        # fit for peak search DM
        print(f"Fitting for peak DM trial (Fitting to a simple quadratic)\n")
        model_samp = trial_dm_vals.size * 50

        peak_samp = np.argmax(trial_dm_vals)
        wind_samp = int(trial_dm_vals.size * 0.05)
        lhs_samp, rhs_samp = wind_samp, wind_samp
        if peak_samp - lhs_samp < 0:
            lhs_samp = peak_samp
        if peak_samp + rhs_samp > trial_dm_vals.size - 1:
            rhs_samp = trial_dm_vals.size - peak_samp - 1

        dm_fit = curve_fit(quadratic, trial_dms[peak_samp - lhs_samp : peak_samp + rhs_samp],
                            trial_dm_vals[peak_samp - lhs_samp : peak_samp + rhs_samp])

        # sn_model = model_curve(trial_dm_vals, samp = model_samp, n = args.n)
        dm_model = np.linspace(trial_dms[peak_samp] - args.DMstep * wind_samp * 2, 
                               trial_dms[peak_samp] + args.DMstep * wind_samp * 2, model_samp)
        sn_model = quadratic(dm_model, *dm_fit[0])

        # get best delDM S/N wise
        args.delDM = dm_model[np.argmax(sn_model)]
    else:
        args.delDM = trial_dms[np.argmax(trial_dm_vals)]

    print(f"Optimal delta DM: {args.delDM:.4f}   [pc/cm^3]")


    # plot trial DM val over DM range
    plt.figure(figsize = (10,10))
    if args.quadfit:
        plt.scatter(trial_dms, trial_dm_vals, c = 'k')
        plt.plot(dm_model, sn_model, 'r')
    else:
        plt.plot(trial_dms, trial_dm_vals, 'k')
    ylim = plt.gca().get_ylim()
    plt.plot([args.delDM]*2, ylim, 'r--')
    plt.xlabel("Trial DM [pc/cm^3]", fontsize = 16)
    plt.ylabel("Trial Score (arb.)", fontsize = 16)
    plt.title(f"Optimal del DM: {args.delDM:.4f}   [pc/cm^3]")
    plt.ylim(ylim)

    plt.savefig("sn_vs_dm.png")
    print("Saving plot of S/N per trial DM as [sn_vs_dm.png]")

    # dedisperse to optimal DM and save as new plot
    dm_shifts = (args.kdm * args.delDM * (1/freq**2 - 1/args.f0**2) / (args.dt * args.tN * 1e-3)).astype(int)
    for i, dm_shift in enumerate(dm_shifts):
        if not flagged_chans[i]:
            dedispersed_dynI[i] = np.roll(dynI[i], -dm_shift)

    maxpos = np.argmax(np.mean(dedispersed_dynI[~flagged_chans], axis = 0))
    maxt = maxpos * (args.dt * args.tN)

    plt.figure(figsize = (10,10))
    plt.imshow(dedispersed_dynI, aspect = 'auto', extent = [0, args.dt * args.nsamp, freq[-1], freq[0]])
    ylim = plt.gca().get_ylim()
    plt.plot([maxt]*2, ylim, 'r--')
    plt.ylim(ylim)
    plt.xlabel("Time [ms]", fontsize = 16)
    plt.ylabel("Frequency [MHz]", fontsize = 16)

    plt.savefig("dedispersed_I.png")
    print(f"Saving plot of best dynamic spectra, de-dispersed by {args.delDM} [pc/cm^3] as [dedispersed_I.png]")

    








def save_output(args):
    """
    De-disperse Dynamic spectrum and save as new file

    """

    ## load data
    with open(args.i, "rb") as file:
        dynI = np.load(file)
    
    args.nchan, args.nsamp = dynI.shape
    args.df = args.bw / args.nchan
    
    ## dedisperse
    dedispersed_dynI = dynI.copy()

    freq = np.linspace(args.cfreq + args.bw/2 - args.df,
                       args.cfreq - args.bw/2 + args.df, args.nchan)
    
    if args.lower:
        freq = freq[::-1]

    args.f0 = np.min(freq)

    dm_shifts = (args.kdm * args.delDM * (1/freq**2 - 1/args.f0**2) / (args.dt * 1e-3)).astype(int)

    # get flagged channels before hand to speed things up
    flagged_chans = np.isnan(dynI[:,0])

    for i, dm_shift in enumerate(dm_shifts):
        if not flagged_chans[i]:
            dedispersed_dynI[i] = np.roll(dynI[i], -dm_shift)


    ## save outputs
    with open(args.o, "wb") as file:
        np.save(file, dedispersed_dynI)

    print(f"Saved dedispersed dynspec as [{args.o}]")







if __name__ == "__main__":
    # main block of code
    
    
    args = get_args()

    # search through DM range and find maximum delta DM
    if args.delDM is None:
        search_DM(args)

    # outputs
    if args.o is not None:
        save_output(args)

    print("[incoherent_dedisperse.py] Complete!!")


