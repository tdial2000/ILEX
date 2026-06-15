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
from ilex.frb import FRB
from ilex.io import ilexIO
from scipy.signal import correlate
import os, sys

if os.path.isdir(os.path.join(os.environ['ILEX_PATH'], "src/SHRINE")):
    from ilex.addons.shrine import get_structure_maximised_dm


def get_args():
    """
    Get arguments
    """

    desc = """ There are two options for performing incoherent dedispersion. 
               1. Manually input stokes I dynspec, with time resolution, downsampling, cfreq and banwdidth 
               2. Pass an ILEX config file (.yaml), this will perform pre-processing before searching for optimal DM. 
                  This is a great option if channel flagging/cropping is nessesary. 

               # Assumptions #
               1. Stokes I dynspec I(f, t) in that order.
               2. First channel is top of the band
               3. Data is real 

               # Methods to search for best DM #
               [simple]: Perform a simple descrete roll per channel using the quadratic relation between DM and frequency and return the highest peak S/N
               [SMDM]: Perform structure maximisation using SHRINE to return the most optimal DM
    """

    parser = argparse.ArgumentParser(description = desc, formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # data input
    parser.add_argument("-i", help = "Stokes I dynamic spectrum (.npy): reference frequency assumed bottom of the band", type = str)
    parser.add_argument("--parfile", help = "ILEX .yaml file for pre-processing", type = str, default = None)
    
    # pre-processing arguments
    parser.add_argument("--dt", help = "Time resolution in ms", type = float, default = 0.001)
    parser.add_argument("--tN", help = "Time averaging factor", type = int, default = 1)
    parser.add_argument("--rfi", help = "Flag coarse RFI channels", action = "store_true")
    parser.add_argument("--rfitN", help = "Downsample factor to apply during RFI flagging", default = 1000, type = int)
    parser.add_argument("--thresh", help = "Threshold for RFI flagging", default = 3.0, type = float)
    # parser.add_argument('--avgfirst', help = "Average data before rolling", action = "store_true")
    
    # DM arguments
    parser.add_argument("--DMmin", help = "Minimum of DM [pc/cm^3] range to search over", type = float, default = -1.0)
    parser.add_argument("--DMmax", help = "Maximum of DM [pc/cm^3] range to search over", type = float, default = 1.0)
    parser.add_argument("--DMstep", help = "Step size of DM [pc/cm^3]", type = float, default = 0.1)
    parser.add_argument("--fref", help = "Reference frequency, options = [min, max, median] for the bottom, top and middle of band. By default min", type = str, default = "min")

    # shrine arguments
    parser.add_argument("--method", help = "Choose method to perform DM search, options are [simple, SMDM], by default 'simple'", type = str, default = "simple")
    parser.add_argument("--force_kc", help = "manually force k index cutoff for low pass filter, if not given, will estimate best value", type = int, default = None)
    parser.add_argument("--filter_order", help = "low pass filter spectral index, by default 3", type = int, default = 3)

    # bandwidth parameters
    parser.add_argument("--cfreq", help = "Central frequency [MHz] of Dynamic spectrum", type = float, default = 919.5)
    parser.add_argument("--bw", help = "Bandwidth [MHz] of Dynamic spectrum", type = float, default = 336)


    # additional arguments
    parser.add_argument("-o", help = "Output filename, No output saved if not specified, if parfile given will add the dsX suffix to each loaded file", type = str, default = None)
    parser.add_argument("--oparfile", help = "Output par file", type = str, default = None)
    parser.add_argument("--delDM", help = "Delta DM [pc/cm^3] to apply for dedispersion, if given will overide search", type = float, default = None)
    parser.add_argument("--quadfit", help = "Fit sn vs dm to a quadratic and extract optimal DM, only for [simple] method", action = "store_true")

    args = parser.parse_args()

    if args.method == "SMDM":
        if not os.path.isdir(os.path.join(os.environ['ILEX_PATH'], "src/SHRINE")):
            print("'SHRINE' has not been installed, install SHRINE in the src/ folder by git cloning from https://github.com/marcinglowacki/SHRINE...")
            sys.exit()

    # add constants
    args.kdm = 4149.377593


    return args



def load_parfile(args):
    """
    
    Process parfile

    """

    frb = FRB(args.parfile)

    args.tN = frb.metapar.tN 
    args.dt = frb.par.dt
    tN = 1
    # if args.avgfirst:
    #     args.tN = 1
    #     args.dt = frb.par.dt * frb.metapar.tN
    #     tN = frb.metapar.tN

    dsI = frb.get_data(['dsI'], get = True, tN = tN)['dsI']

    args.bw = frb.this_par.bw
    args.cfreq = frb.this_par.cfreq



    return dsI




def load_data(args):
    """ Load data

    Alteres args

    Returns
    -------
    dynI: 2D np.ndarray
        Stokes I dynspec I(f, t)
    freq: 1D np.ndarray
        frequency datas [MHz]
    """

    # load in data and create freq array
    if args.parfile is not None:
        dynI = load_parfile(args)
    else:
        with open(args.i, "rb") as file:
            dynI = np.load(file)
    
    args.nchan, args.nsamp = dynI.shape
    args.df = args.bw / args.nchan

    freq = np.linspace(args.cfreq + args.bw/2 - args.df/2,
                       args.cfreq - args.bw/2 + args.df/2, args.nchan)
    
    # if args.lower:
    #     freq = freq[::-1]

    if args.fref == "min":
        args.f0 = np.min(freq)
    elif args.fref == "max":
        args.f0 = np.max(freq)
    elif args.fref == "median":
        args.f0 = np.median(freq)
    else:
        ValueError(f"--fref = [{args.fref}] not supported!")
    print(f"Using [{args.fref}] f0: {args.f0}")

    if args.method == "SMDM":
        method = "[SMDM] Structure maximising using SHRINE"
    elif args.method == "WMDM":
        method = "[WMDM] Minimising burst width"
    elif args.method == "simple":
        method = "[simple] S/N Maximising using descrete rolling"
    else:
        ValueError(f"method [{args.method}] not supported, see -h for options...")

    print(f"Performing De-dispersion: {method}")
    print("="*25, "\n")
    print(f"dt: {args.dt} [ms] -> {args.tN} * {args.dt} = {args.tN * args.dt} [ms]")
    print("cfreq:".ljust(15) +  f"{args.cfreq} [MHz]")
    print("bw:".ljust(15) + f"{args.bw} [MHz]\n")
    print("Min DM  |  Max DM  |  DM step    [pc/cm^3]")
    print("-"*29)
    print(f"{args.DMmin:.4f}".ljust(8) + "| " + f"{args.DMmax:.4f}".ljust(9)  + "| " + f"{args.DMstep:.4f}\n")

    return dynI, freq





def generate_profiles(ds, dt, tN, cfreq, bw, f0, dms):
    """
    Generate I matrix, a matrix of Stokes I time series for each trial DM.
    This function is an incoherent alternative to the native profile generation avaliable
    in SHRINE.

    Parameters
    ----------
    ds: 2D np.ndarray
        Stokes I dynamic spectrum
    dt: float
        sample resolution [ms]
    tN: int
        Downsampling factor in time
    cfreq: float
        Central frequency [MHz]
    bw: float
        Bandwidth [MHz]
    f0: float
        Reference frequency
    dms: 1D np.ndarray or array-like
        List of trial DMs
    
    Returns
    -------
    I: 2D np.ndarray
        I matrix (dm, t)
    """
    kdm = 4149.377593

    I = np.zeros((dms.size, ds.shape[1]//tN), dtype = float)

    # create freq array
    df = ds.shape[0] / bw
    freqs = np.linspace(cfreq + bw/2 - df/2, cfreq - bw/2 + df/2,
                        ds.shape[0])

    # Create copy of ds to dedisperse by rolling
    ds_dedis = ds.copy()

    # check if any channels are flagged
    goodchans = np.where(~np.isnan(ds[:, 0]))[0]

    # loop through trial dms and get dedispersed time series
    for i, dm in enumerate(dms):
        print(f"Searching trial DMs...   {i/dms.size:.2%}", end = "\r")

        # calculate dm shifts
        dmshifts = (kdm * dm * (1/freqs**2 - 1/f0**2)/(dt * 1e-3)).astype(int)

        # dedisperse by rolling
        for _, chan in enumerate(goodchans):
            ds_dedis[chan] = np.roll(ds[chan], -dmshifts[chan])
        
        # scrunch in frequency
        I[i] = average(np.mean(ds_dedis[goodchans], axis = 0), N = tN)
    print("Searching trial DMs...   100.00%")
    print("Trial DM profiles Generated")

    return I





def search_DM(args):
    """
    Incoherently search through DM range 

    """

    # get data
    dynI, freq = load_data(args)
    
    # flag channels of RFI if applicable
    if args.rfi:
        flagchan = medrms_chanflag(dynI, args.thresh, args.rfitN)
        dynI[flagchan] = np.nan

    # pre-process
    tN = args.tN
    # if args.avgfirst:
    #     dynI = average(dynI, N = args.tN, axis = 1)
    #     tN = 1

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
        
        dm_shifts = (args.kdm * trial_dm * (1/freq**2 - 1/args.f0**2) / (args.dt * args.tN / tN * 1e-3)).astype(int)

        for j, dm_shift in enumerate(dm_shifts):
            if not flagged_chans[j]:
                dedispersed_dynI[j] = np.roll(dynI[j], -dm_shift)
        
        # scrunch to get time series and find maximum of time series
        trial_dm_vals[i] = np.max(np.nanmean(average(dedispersed_dynI[~flagged_chans], N = tN, axis = 1), axis = 0))
    
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
    dm_shifts = (args.kdm * args.delDM * (1/freq**2 - 1/args.f0**2) / (args.dt * args.tN / tN * 1e-3)).astype(int)
    for i, dm_shift in enumerate(dm_shifts):
        if not flagged_chans[i]:
            dedispersed_dynI[i] = np.roll(dynI[i], -dm_shift)

    dedispersed_dynI = average(dedispersed_dynI, N = tN, axis = 1)

    maxpos = np.argmax(np.nanmean(dedispersed_dynI[~flagged_chans], axis = 0))
    maxt = maxpos * (args.dt * args.tN)

    x = np.linspace(args.dt/2, args.dt/2 + (args.nsamp-1) * args.dt, dedispersed_dynI.shape[1])

    fig, ax = plt.subplots(2, 1, figsize = (10,10), gridspec_kw = {'height_ratios':[1, 4]}, sharex = True)
    ax = ax.flatten()
    ax[1].imshow(dedispersed_dynI, aspect = 'auto', extent = [0, args.dt * args.nsamp, freq[-1], freq[0]])
    ylim = ax[1].get_ylim()
    ax[1].plot([maxt]*2, ylim, 'r--')
    ax[1].set_ylim(ylim)
    ax[1].set_xlabel("Time [ms]", fontsize = 16)
    ax[1].set_ylabel("Frequency [MHz]", fontsize = 16)

    # time series plot
    ax[0].plot(x, np.nanmean(dedispersed_dynI, axis = 0), 'k', linewidth = 2)
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_ylabel("Flux Density (arb)", fontsize = 16)
    ylim = ax[0].get_ylim()
    ax[0].plot([maxt]*2, ylim, 'r--')
    ax[0].set_ylim(ylim)

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)
    

    plt.savefig("dedispersed_I.png")
    print(f"Saving plot of best dynamic spectra, de-dispersed by {args.delDM} [pc/cm^3] as [dedispersed_I.png]")

    







def search_SMDM(args):
    """
    Get best delta DM through structure maximisation
    
    """

    # get data
    dynI, _ = load_data(args)

    if args.rfi:
        flagchan = medrms_chanflag(dynI, args.thresh, args.rfitN)
        dynI[flagchan] = np.nan

    # where to save files and figures, make a seperate directory for these since
    # there are so many.
    outdir = os.path.join(os.getcwd(), "SMDM")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    label = "SMDM"
    if args.o is not None:
        label = args.o

    args.delDM = get_structure_maximised_dm(ds = dynI, dt = args.dt, tN = args.tN,
                                            cfreq = args.cfreq, bw = args.bw, f0 = args.f0, DMmin = args.DMmin,
                                            DMmax = args.DMmax, DMstep = args.DMstep, label = label, 
                                            outdir = outdir, filter_order = args.filter_order, 
                                            force_kc = args.force_kc)
    

    return








def medrms_chanflag(dsI, threshold = 3.0, tN = 1000):
    """
    Flag channels in dynamic spectrum by filtering out RFI that exceeds some threshold given by
    threshold * np.median(np.abs(np.std(I))) where np.median(np.std(I)) is subtracted from np.std(I).

    Parameters
    ----------
    dsI : 2D np.ndarray
        Stokes I dynamic spectrum (f,t)
    threshold : float
        threshold value 
    tN : int
        Downsampling to apply to data before coarse RFI flagging

    Returns
    -------
    flagchans : 1D np.ndarray
        Array of channel indicies where RFI was detected
    
    """

    # std and median calculations
    stdfI = np.nanstd(average(dsI, axis = 1, N = tN), axis = 1)
    stdfI = np.abs(stdfI - np.nanmedian(stdfI))
    flagchan = np.where(stdfI > (threshold * np.nanmedian(stdfI)))[0]

    return flagchan











def save_output(args, stk = "I"):
    """
    De-disperse Dynamic spectrum and save as new file

    """

    ## load data
    if args.parfile is not None:
        frb = FRB(args.parfile)
        dynI = np.load(frb.get_filepaths(stk))

        args.dt = frb.par.dt
        args.bw = frb.par.bw
        args.cfreq = frb.par.cfreq

    else:
        with open(args.i, "rb") as file:
            dynI = np.load(file)
    
    args.nchan, args.nsamp = dynI.shape
    args.df = args.bw / args.nchan
    
    ## dedisperse
    dedispersed_dynI = dynI.copy()

    freq = np.linspace(args.cfreq + args.bw/2 - args.df/2,
                       args.cfreq - args.bw/2 + args.df/2, args.nchan)
    
    # if args.lower:
    #     freq = freq[::-1]

    if args.fref == "min":
        args.f0 = np.min(freq)
    elif args.fref == "max":
        args.f0 = np.max(freq)
    elif args.fref == "median":
        args.f0 = np.median(freq)
    else:
        ValueError(f"--fref = [{args.fref}] not supported!")

    dm_shifts = (args.kdm * args.delDM * (1/freq**2 - 1/args.f0**2) / (args.dt * 1e-3)).astype(int)

    # get flagged channels before hand to speed things up
    flagged_chans = np.isnan(dynI[:,0])

    for i, dm_shift in enumerate(dm_shifts):
        if not flagged_chans[i]:
            dedispersed_dynI[i] = np.roll(dynI[i], -dm_shift)


    ## save outputs
    filename = args.o
    if args.parfile is not None:
        filename += f"_ds{stk}.npy"
    with open(filename, "wb") as file:
        np.save(file, dedispersed_dynI)

    print(f"Saved dedispersed dynspec as [{filename}]")





def save_parfile(args):

    

    if args.parfile is not None:
        frb = FRB(args.parfile)
        datafilepath = args.o
        overwrite = False
    else:
        frb = FRB(cfreq = args.cfreq, bw = args.bw, dt = args.dt,
                  tN = args.tN, DM = args.delDM)
        frb.load_data(dsI = args.o)
        overwrite = True


    ilexio = ilexIO(filepath = args.oparfile, frb = frb, 
                datafilepath = datafilepath, overwrite = overwrite)
    ilexio.save()








if __name__ == "__main__":
    # main block of code
    
    
    args = get_args()

    # search through DM range and find maximum delta DM
    if args.delDM is None:
        if args.method == "SMDM":
            search_SMDM(args)
        elif args.method == "simple":
            search_DM(args)
        else:
            ValueError(f"method [{args.method}] not supported, see -h for options...")

    # outputs
    if args.o is not None:
        if args.parfile is not None:
            frb = FRB(args.parfile)
            for s in "IQUV":
                if frb.ds[s] is not None:
                    save_output(args, s)
        else:
            save_output(args)

    # save parfile
    if (args.oparfile is not None) and (args.o is not None):
        print("Saving output as new ILEX config file!")
        save_parfile(args)


    print("[incoherent_dedisperse.py] Complete!!")


