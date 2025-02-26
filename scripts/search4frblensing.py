##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     20/02/2025                 #
# Date (updated):     20/02/2025                 #
##################################################
#                                                #
#                                                #
# Search for lensing in FRB voltage data         #
##################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.fft import fft, next_fast_len
import argparse
from ilex.data import average, acf
from math import ceil
import sys, warnings, os
from make_dynspec import make_ds, baseline_correction, flag_chan



class _empty:
    pass



#############################################

# Utility functions

#############################################

def apply_gaussian_smooth(y, stDev: int = 3):
    """
    Apply a gaussian as a smoothing function to [y]

    Parameters
    ----------
    y : np.ndarray or array-like
        Data to smooth
    stDev : int
        Standard deviation of gaussian in number of samples
    """

    x = np.linspace(-6*stDev, 6*stDev, 12*stDev + 1)
    gaussian = np.exp(-x**2/(2*stDev**2))

    return np.convolve(y, gaussian/np.sum(gaussian), mode = "same")


def get_crop(args, data, nFFT: int = 336):
    """
    Get crop by making dynamic spectrum

    Parameters
    ----------
    X : complex X data
    Y : complex Y data
    """

    datapath = os.path.join(args.o, args.dsI)

    temp_args = _empty()
    temp_args.do_chanflag = True

    make_new_dsI = False
    if args.dsI is None:
        make_new_dsI = True
    if datapath is not None:
        if not os.path.isfile(datapath):
            make_new_dsI = True

    if make_new_dsI:
        # load in data
        X = np.load(args.x, "r")
        Y = np.load(args.y, "r")

        print(f"Loaded [X] data from: {args.x}")
        print(f"Loaded [Y] data from: {args.y}")

        # make dynspec
        I = make_ds(X, Y, nFFT = nFFT)
        I[0] *= 1e-12

        if args.dsI is not None:
            with open(datapath, 'wb') as file:
                print(f"Saving full Stokes I dynspec as [{datapath}] for quick loading later...")
                np.save(file, I)

    else:
        print(f"Quick loading full Stokes I dynspec from [{datapath}]...")
        I = np.load(datapath, 'r')

    I_raw = I.copy()
    I_raw -= np.mean(I_raw, axis = 1)[:, None]

    chanflag,_ = flag_chan(I_raw, 10, 1000, temp_args, None)

    # get bounds
    I_raw[chanflag] = np.nan
    
    data['tIfull'] = average(np.nanmean(I_raw, axis = 0), N = args.tN)

    # parameters for on pulse and off pulse crop
    peak_samp = int(np.argmax(data['tIfull']) * args.tN * 336)
    width_samp = int(args.t / args.idt / 2)

    rms_g_samp = int(args.rms_g / args.idt)
    rms_w_samp = int(args.rms_w / args.idt / 2)

    X = np.load(args.x, mmap_mode = "r")
    Y = np.load(args.y, mmap_mode = "r")

    onpulse_window = slice(peak_samp - width_samp,peak_samp + width_samp)
    offpulse_window = slice(peak_samp - width_samp - rms_g_samp - rms_w_samp,
                            peak_samp - width_samp - rms_g_samp)

    # crop
    data['X'], data['Y'] = X[onpulse_window], Y[onpulse_window]
    data['Xerr'], data['Yerr'] = X[offpulse_window], Y[offpulse_window]

    args.peak_samp = peak_samp
    args.onpulse_window = onpulse_window
    args.offpulse_window = offpulse_window


    return






def get_flaggedchans(chanflag, nFFT: int = 336):
    """
    Get channels to flag from str

    Parameters
    ----------
    chanflag : str
        channels to flag (indicies)


    """

    chans2flag = chanflag.split(',')
    flaggedchans = []
    for i, chans in enumerate(chans2flag):
        if ":" in chans:
            lchan = int(float(chans.split(':')[0]) * (nFFT-1))
            rchan = int(float(chans.split(':')[1]) * (nFFT-1))
            flaggedchans += list(range(lchan,rchan+1))
        
        else:
            chan = int(float(chans) * (nFFT-1))
            flaggedchans += [chan]

    
    return np.abs(np.array(flaggedchans) - nFFT + 1)





def auto_channel_flag(dsI, flag_thresh, tN):
    """
    Use off pulse crop to do auto channel flagging
    """

    # create boolean array
    chanmask = np.arange(0, dsI.shape[0]).astype(int)

    ds_avg = average(dsI, axis = 1, N = tN, nan = True)
    f_std = np.nanstd(ds_avg, axis = 1)
    med_rms = np.nanmedian(f_std)
    mad_rms = 1.48 * np.nanmedian(np.abs(f_std - med_rms))

    chan2flag = np.where(f_std > (med_rms + flag_thresh*mad_rms))[0]
    return chanmask[chan2flag]




def get_args():

    desc = """
    Search for Lensing in FRB voltage data. Requires complex voltage data X and Y.
    You can provide a list of channelisations i.e. --nFFT "336,512,1024,1680" will perform 4 different
    ACFs with 336, 512, 1024 and 1680 channels respectively.
    """

    parser = argparse.ArgumentParser(description = desc)
    parser.add_argument("-x", help = "X voltage data", required = True, type = str)
    parser.add_argument("-y", help = "Y voltage data", required = True, type = str)

    parser.add_argument("--nFFT", help = "channelisation - Number of channels, can provide a list", type = str, default = "336")
    parser.add_argument("-t", help = "Window length of dynspec in time [ms] centered at maximum of burst", default = 100.0, type = float)

    # baseline arguments (for correction)
    parser.add_argument("--rms_w", help = "Width of rms region for S/N estimation [ms]", type = float, default = 20.0)
    parser.add_argument("--rms_g", help = "Width of window seperating on pulse and off pulse windows (guard) [ms]", type = float, default = 10.0)
    parser.add_argument("--stDev", help = "stDev of gaussian smoothing filter in samples", type = int, default = 3)
    parser.add_argument("--Wsigma", help = "S/N threshold for zeroing baseline in weight function [if set to zero, will not zero any weights]", type = float, default = 3.0)

    # optional variables
    parser.add_argument('--idt', help = "intrinsic time resolution [ms]", default = 0.00000297619, type = float)
    parser.add_argument('--idf', help = "intrinsic freq resolution [MHz]", default = 1.0, type = float)
    parser.add_argument('--tN', help = "Averaging factor to help with finding peak", default = 100, type = int)
    parser.add_argument("--chanflag", help = "Channels to flag [give indicies]", default = None, type = str)

    # plotting paramters, does not affect acf
    parser.add_argument("--showplots", help = "Show interactive plots", action = "store_true")
    parser.add_argument("--dsI", help = "Raw dynspec file for initial cropping. If no file is found, will make dynspec", type = str, default = None)
    parser.add_argument("-o", help = "Output dir", type = str, default = os.getcwd())

    args = parser.parse_args()


    # format args.nFFT in case of list given
    args.nFFT = np.asarray(args.nFFT.split(','), dtype = int)

    return args







def load_data(args):
    """
    
    Load data and perform cropping

    Parameters
    ----------
    args : input arguments
    """

    data = {}
    data['X'] = None        # On pulse window of complex voltage X
    data['Y'] = None        # On pulse window of complex voltage Y
    data['Xerr'] = None     # Off pulse window of comples voltage X
    data['Yerr'] = None     # Off pulse window of comples voltage Y
    data['dsI'] = None      # On pulse window of Stokes I dynspec
    data['dsX'] = None      # On pulse complex X dynspec
    data['dsY'] = None      # On pulse complex Y dynspec
    data['dsIerr'] = None   # Off pulse window of Stokes I dynspec
    data['tI'] = None       # On pulse window of Stokes I time series
    data['tIfull'] = None   # Full Stokes I time series
    data['tIerr'] = None    # RMS (Variance) in time samples
    data['tImf'] = None     # tI matched filter - Smoothed Gaussian Power times series
    data['acf'] = None      # Channelised auto-correlation voltage dynspec

    # load in data
    get_crop(args, data, nFFT = 336)

    return data





def do_acf(args, data, nFFT: int = 336):
    """
    Do auto correlation with matched filter imaging

    Parameters
    ----------
    args : input arguments
    data : Dictionary of data 
    """
    print(f"[Channelising to {nFFT} Channel]")
    print("-"*25)

    def normalized_complex_acf(x):
        x_power = np.abs(x)**2
        with warnings.catch_warnings(): # ignore zero division errors (they are inevitable - quote that purple guy from Avengers end game)
            warnings.simplefilter("ignore")
            acf = correlate(x, x) / correlate(x_power, x_power)**0.5
        return acf

    nsamp = int(data['X'].size // nFFT)

    # make stokes I dynspec
    data['dsI'] = make_ds(data['X'], data['Y'], nFFT = nFFT)
    data['dsIerr'] = make_ds(data['Xerr'], data['Yerr'], nFFT = nFFT)

    # baseline correction
    data['dsI'] -= np.mean(data['dsIerr'], axis = 1)[:, None]
    data['dsIerr'] -= np.mean(data['dsIerr'], axis = 1)[:, None]

    # flag
    auto_flaggedchans = auto_channel_flag(data['dsIerr'], 10, 1000)
    flaggedchans = np.array([], dtype = int)
    if args.chanflag is not None:
        flaggedchans = get_flaggedchans(args.chanflag, nFFT)

    # average dynspec
    data['dsI'] = average(data['dsI'], axis = 1, N = args.tN)
    data['dsIerr'] = average(data['dsIerr'], axis = 1, N = args.tN)

    # combine flagging
    flaggedchans = np.unique(np.append(auto_flaggedchans, flaggedchans))

    data['dsI'][flaggedchans] = np.nan
    data['dsIerr'][flaggedchans] = np.nan

    # get weight function
    data['tI'] = np.nanmean(data['dsI'], axis = 0)
    data['tIerr'] = np.std(np.nanmean(data['dsIerr'], axis = 0))

    gaussian_weights = apply_gaussian_smooth(data['tI'], args.stDev)
    data['tImf'] = np.interp(np.linspace(0, 1.0, nsamp),
                    np.linspace(0, 1.0, gaussian_weights.size), gaussian_weights)
    
    if args.Wsigma > 0.0:
        data['tImf'][data['tImf'] < args.Wsigma * data['tIerr']] = 0.0  # zero noise in Weight function
    else:
        print(f"No threshold flagging in weight function will be done as Wsigma = {args.Wsigma}")


    # Build X and Y complex dynspec
    data['acf'] = np.zeros((nFFT, 2*nsamp - 1))
    data['dsX'] = np.zeros((nFFT, nsamp), dtype = data['X'].dtype)
    data['dsY'] = np.zeros((nFFT, nsamp), dtype = data['X'].dtype)
    for i in range(nsamp):

        if not (i % 50):
            print(f"[nFFT = {nFFT}] Making Stokes X and Y complex dynspec: {i/nsamp:.2%}", end = "\r")

        wind = slice(i*nFFT,(i+1)*nFFT)
        data['dsX'][:, i] = fft(data['X'][wind])
        data['dsY'][:, i] = fft(data['Y'][wind])

    print(f"[nFFT = {nFFT}] Making Stokes X and Y complex dynspec: 100.00%")

    
    # construct ACF dynspec
    for i in range(nFFT):
        print(f"[nFFT = {nFFT}] Making ACF dynspec: {i/nFFT:.2%}", end = "\r")
        data['acf'][i, :] = (np.abs(normalized_complex_acf(data['dsX'][i, :]*data['tImf']))**2 +
                             np.abs(normalized_complex_acf(data['dsY'][i, :]*data['tImf']))**2)
    print(f"[nFFT = {nFFT}] Making ACF dynspec: 100.00%")

    # flag acf
    data['acf'][flaggedchans] = np.nan

    return



def clean_acf(acf):
    """
    Clean acf

    Parameters
    ----------
    acf :np.ndarray or array-like
        2D acf dynspec
    """

    cleaned_acf = acf.copy()

    # remove zero lag peak and surrounding samples due to PFB response
    cleaned_acf[:, acf.shape[1]//2 - 50 : acf.shape[1]//2 + 51] = np.nan

    # clean near zero time samples
    tI_acf = np.nanmean(cleaned_acf, axis = 0)

    # clean inf
    tI_acf[np.abs(tI_acf) == np.inf] = 0.0

    baseline = np.nanmean(tI_acf)
    print(baseline, 0.01*baseline)
    cleaned_acf[:, tI_acf < 0.01*baseline] = np.nan


    return cleaned_acf









def plot_diagnostics(args, data, nFFT):
    """
    Diagnostics plotting
    """

    print(f"[nFFT = {nFFT}] Making plots\n")

    ################################################################
    # Make figure comparing Stokes I dynspec and ACF power dynspec
    ################################################################
    fig, ax = plt.subplots(2, 2, figsize = (14,10), gridspec_kw = {'height_ratios':[1, 3]})
    ax = ax.flatten()

    x = np.linspace(-args.t / 2, args.t / 2, data['tI'].size)

    ds_acf = clean_acf(data['acf'])

    # Power time series
    ax[0].plot(x, data['tI'], 'k')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title("Intensity")

    # Stokes I dynspec
    ax[2].imshow(data['dsI'], aspect = 'auto', extent = [-args.t / 2, args.t / 2, 0, 1])
    ax[2].set_xlabel("Time [ms]", fontsize = 16)
    ax[2].set_ylabel("normalised chans", fontsize = 16)
    ax[2].sharex(ax[0])

    tI_acf = np.nanmean(ds_acf, axis = 0)
    x_acf = np.linspace(-args.t, args.t, tI_acf.size)

    # ACF time series
    ax[1].plot(x_acf, tI_acf, 'k')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title("Time-lag (ACF)")

    # ACF dynspec
    ax[3].imshow(ds_acf, aspect = 'auto', extent = [-args.t, args.t, 0, 1])
    ax[3].set_xlabel("Time-lag [ms]", fontsize = 16)
    ax[3].get_yaxis().set_visible(False)
    ax[3].sharex(ax[1])
    ax[3].sharey(ax[2])

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(os.path.join(args.o, f"lensing_{nFFT}_acf.png"))




    ################################################################
    # Plot Gaussian smoothing function
    ################################################################

    fig2, ax2 = plt.subplots(2, 1, figsize = (12,10))
    ax2 = ax2.flatten()

    ax2[0].plot(data['tIfull'], 'k')
    ylim = ax2[0].get_ylim()
    ax2[0].plot([args.peak_samp / args.tN / 336]*2, ylim, '--r', label = "Burst peak")
    ax2[0].plot([args.onpulse_window.start / args.tN / 336]*2, ylim, '--b', label = "On pulse window")
    ax2[0].plot([args.onpulse_window.stop / args.tN / 336]*2, ylim, '--b')
    ax2[0].plot([args.offpulse_window.start / args.tN / 336]*2, ylim, '--m', label = "Off pulse window")
    ax2[0].plot([args.offpulse_window.stop / args.tN / 336]*2, ylim, '--m')
    ax2[0].set_ylim(ylim)
    ax2[0].get_xaxis().set_visible(False)
    ax2[0].set_ylabel("Flux Density (arb.)", fontsize = 16)
    ax2[0].legend(loc = "upper right", fancybox = True)



    ax2[1].plot(x, data['tI'], 'k')
    ax2[1].plot(np.linspace(-args.t / 2, args.t / 2, data['tImf'].size), data['tImf'], '--r')
    ax2[1].set_xlabel("Time [ms]", fontsize = 16)
    ax2[1].set_ylabel("Flux Density (arb.)", fontsize = 16)
    xlim = ax2[1].get_xlim()

    for i, sig in enumerate([1, 2, 3, 5, 10, 20]):
        ax2[1].plot(xlim, [sig * data['tIerr']]*2, '--', label = f"{sig}")
    ax2[1].set_xlim(xlim)
    
    ax2[1].legend(title = "S/N threshold", fancybox = True, loc = "upper right")

    fig2.tight_layout()
    fig2.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(os.path.join(args.o, f"lensing_{nFFT}_crop.png"))


    return










def plot_different_acfs(args, acfs):
    """
    Plot acfs with different channelisations and compare

    Parameters
    ----------
    args : argparse.args
        inputs
    acfs : dict
        time series of different channelised acfs
    """

    plt.figure(figsize = (10,12))

    maxs = []
    for i, nfft in enumerate(acfs.keys()):
        acfs[nfft][np.abs(acfs[nfft]) == np.inf] = np.nan
        acfs[nfft][np.isnan(acfs[nfft])] = 0.0
        maxs += [np.max(acfs[nfft])]

    for i, nfft in enumerate(acfs.keys()):
        tI_acf = acfs[nfft] / max(maxs)
        x_acf = np.linspace(-args.t, args.t, tI_acf.size)

        plt.plot(x_acf, tI_acf + i, label = f"nFFT = {nfft}")

    plt.legend(title = "Number of channels [nFFT]", loc = "upper right", 
                fancybox = True)
    plt.xlabel("Time-lag [ms]", fontsize = 16)
    plt.gca().get_yaxis().set_visible(False)

    plt.savefig(os.path.join(args.o, "lensing_nFFT_list.png"))

    return


    
        









if __name__ == "__main__":

    print("#"*32)
    print("Searching for lensing in FRB...")
    print("#"*32)

    args = get_args()

    # load data and crop
    data = load_data(args)

    ACFS = {}
    for _, nFFT in enumerate(args.nFFT):

        # perform auto correlation 
        do_acf(args, data, nFFT)

        # plot
        plot_diagnostics(args, data, nFFT)

        if len(args.nFFT) > 1:
            # delete figures
            for i in plt.get_fignums():
                plt.close(i)

            ACFS[str(nFFT)] = np.nanmean(data['acf'], axis = 0)

    if len(args.nFFT) > 1:
        # make channelise group plot
        print("Plotting diagnostics for the different channelisations...")
        plot_different_acfs(args, ACFS)

    print("search4frblensing.py Completed!\n")

    if args.showplots:
        plt.show()


    # END OF SCRIPT