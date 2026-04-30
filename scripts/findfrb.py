# imports 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ilex.frb import FRB 
from ilex.data import average, get_zapstr, gaussian_smooth
from ilex.widths import find_optimal_fluence_width
from ilex.io import ilexIO
import argparse, os

# testing
from scipy.optimize import curve_fit


#### THINGS TODO ####
# Add verbose plotting option -v


class _empty_class:
    pass


_findfrb_globals = _empty_class()
_findfrb_globals._valid_fcrop_methods = ["minfluence", "fsig"]


def get_args():
    desc = """
        Locate time and frequency position of FRB. Following steps:

        1. Coarse RFI channel flagging using standard deviation median threshold
        2. Approximate width (temporal bounds) of burst by finding the minimum burst width that encompasses X% of fluence
        3. Apply boxcar/gaussian kernel matched filter to FRB to get spectrum. 
        4. Approximate spectral bounds of burst using either minimum burst width fluence or sigma threshold
        4. Using initial temporal and spectral bounds, Perform RFI subtraction
        5. Repeat coarse RFI channel flagging step on raw RFI subtracted data
        6. Repeat Steps 2 and 3 to refine temporal and spectral bounds of FRB
        7. Save RFI subtracted, RFI flagged dataset with metadata .txt file containing burst parameters. 

        if an ILEX config file is provided through the --parfile option, FRB parameters such as --dt, --cfreq and --bw will be taken
        from said config file.
        Only the stokes I dynspec will be loaded in when searching for the FRB. However, the RFI subtractions and RFI flagging will be 
        performed on all Stokes dynspec files defined in the ILEX config file.

        You can save these updated data products by specifying the --save_data option.

        Any new Stokes dynspec files saved will have the filename <-o>_ds<S>.npy, where -o is the script option and <S> is the stokes parameter.
        An ILEX config file will always be created.

        [Caution!] - If you rewrite your ILEX config file, all the unused parameters will be reset!

        The user also has the option of saving only a crop of the dynamic spectrum using the --cropds option. For example --cropds=200.0 will save 
        a crop of the data that is centered on the located FRB and is 200.0 ms wide. This is useful to reduce the amount of disk space being used.
        This will only apply when the --save_data option is specified.

    """

    parser = argparse.ArgumentParser(description = desc, 
                        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # inputs
    parser.add_argument("-i", help = "Stokes I dynamic spectrum (f,t)", type = str)
    parser.add_argument("--parfile", help = "Input ILEX config file", default = None, type = str)

    # Coarse RFI flagging parameters
    parser.add_argument("--tN", help = "Time Downsampling factor applied to data when performing coarse RFI channel flagging and for peak searching", 
                        default = 1000, type = int)
    parser.add_argument("--thresh", help = "Threshold for coarse RFI flagging", default = 3.0, type = float)
    parser.add_argument("--rfiter", help = "Number of iterations to perform when doing coarse RFI flagging", default = 1, type = int)
    parser.add_argument("--tcrop", help = "Time crop to search over when looking for peak of burst [min, max] in [milliseconds]", nargs = 2, type = float, default = None)


    # Fluence width parameters
    parser.add_argument("--yfrac", help = "Fraction of total fluence of burst for minimum width to encompass", default = 0.95, type = float)

    # Buffer parameters
    parser.add_argument("-w", help = "Width of window [milliseconds] to search for FRB burst centered on peak in dynamic spectrum",
                        default = 100.0, type = float)

    # spectrum flagging parameters
    parser.add_argument("--fsig", help = "Sigma threshold to apply to spectrum to get bounds", default = 3.0, type = float)

    rmsw_desc = """rms window on LHS of burst [milliseconds], used along with --fsig. If LHS samples are too few, will attempt the RHS. This option is also used for rfi subtraction,
                   where half the width on either side of the burst will be taken and used to subtract slow varying RFI. The RFI windows taken will have a gap between them and the burst, i.e
                   the guard, which is equal to --rmsw/2."""

    parser.add_argument("--rmsw", help = rmsw_desc, default = 10.0, type = float)

    fcrop_method_desc = """Method used to determine crop of Burst in frequency. There are two options:
                            1. [fsig]: --fsig threshold flagging -> Uses bounds of channels that meet this threshold
                            2. [minfluence]: find minimum burst width that encompasses of burst fluence in Frequency"""

    parser.add_argument('--fcrop_method', help = fcrop_method_desc, default = "minfluence", type = str)
    parser.add_argument("--rfisub", help = "Perform RFI subtraction. --rmsw will be used.", action = "store_true")

    stdev_desc = """Standard deviation in number of samples for gaussian smoothing kernel applied to time data, the result of which will be used
                    as a on-pulse matched filter when creating frequency spectrum and finding [fcrop]. Set --stDev=0 to disable feature and apply simple boxcar filter."""

    parser.add_argument("--stDev", help = stdev_desc, default = 1, type = int)

    # FRB parameters
    parser.add_argument("--dt", help = "Sample resolution [milliseconds]", default = 1e-3, type = float)
    parser.add_argument("--cfreq", help = "Central frequency of observation [MHz]", default = 919.5, type = float)
    parser.add_argument("--bw", help = "Bandwidth of observation [MHz]", default = 336.0, type = float)
    parser.add_argument("--Nband", help = "Number of sub-bands to split up burst for sub-banded searching, set to 0 to disable this feature!", default = 0, type = int)
    
    # outputs
    parser.add_argument("-o", help = "Output prefix for files and images", default = "findfrb", type = str)
    parser.add_argument("--save_data", help = "Save copy of stokes dynspec file/s with zapped RFI channels", action = "store_true")
    parser.add_argument("--oparfile", help = "Output ILEX yaml file", default = "findfrb.yaml", type = str)
    parser.add_argument("--cropds", help = "Crop full dynspec and only save a certain region [--cropds] centered on the located FRB [milliseconds]", default = None, type = float)

    # plotting
    parser.add_argument("-p", help = "Plot results", action = "store_true")
    parser.add_argument("-v", help = "Plot MORE results", action = "store_true")
    parser.add_argument("--pw", help = "Width of dynspec [for plotting only]", type = float, default = 200.0)
    parser.add_argument("--pfN", help = "Downsampling factor in frequency to apply when plotting dynspec", type = int, default = 1)

    args = parser.parse_args()

    # check args
    if args.fcrop_method not in _findfrb_globals._valid_fcrop_methods:
        ValueError(f"--fcrop_method option must be one of the following: {_findfrb_globals._valid_fcrop_methods}")

    if args.tcrop is None:
        args.tcrop = [0, 'max']

    return args








def load_data(args):

    if args.parfile is not None:
        frb = FRB(args.parfile)
        args.i = frb._data_files['dsI']
        args.dt, args.cfreq, args.bw = frb.par.dt, frb.par.cfreq, frb.par.bw 
    return











def main(args):
    """
    Main script to search for bounds of FRB along with pre-processing routines
    
    """

    # initialise variables
    outputs = {}

    # load data
    frb = FRB(cfreq = args.cfreq, bw = args.bw, dt = args.dt)
    frb.load_data(dsI = args.i)
    frb.set(df = args.bw/frb.par.nchan)
    dsI = frb.get_data("dsI", tN = args.tN, get = True)['dsI']

    # initial coarse RFI flagging and peak finding
    flagchan, outputs['stdfI_1'] = medrms_chanflag(dsI, args.thresh, 1, args.rfiter)
    dsI = frb.get_data("dsI", tN = args.tN, t_crop = args.tcrop, get = True)['dsI']
    outputs['full_xextent'] = frb.this_par.t_lim.copy()
    outputs['flagchan_1'], outputs['dsI_full'] = flagchan.copy(), dsI.copy()
    dsI[flagchan] = np.nan 

    # perform subband search if enabled
    if args.Nband > 1:
        peaksamp, outputs['fcrop_0'] = subband_search(args, outputs, dsI)
    else:
        peaksamp = np.argmax(np.nanmean(dsI, axis = 0))
        outputs['fcrop_0'] = ['min', 'max']
    
    peakpos = peaksamp * args.dt * args.tN + frb.this_par.t_lim[0]
    windowcrop = [peakpos - args.w/2, peakpos + args.w/2]
    centroid = 0
    # centroid, windowcrop = center_fluence(args, frb, flagchan, windowcrop)
    outputs['windowcrop'] = windowcrop.copy()
    
    # check window crop bounds
    if outputs['windowcrop'][0] < frb.par.t_lim[0]:
        outputs['windowcrop'][0] = frb.par.t_lim[0]
    if outputs['windowcrop'][1] > frb.par.t_lim[1]:
        outputs['windowcrop'][1] = frb.par.t_lim[1]

    outputs['peaksamp'], outputs['peakpos'], outputs['centroid'] = peaksamp, peakpos, centroid

    # get data for first run of find_frb
    data = frb.get_data("dsI", t_crop = outputs['windowcrop'], tN = 1, get = True)
    data['dsI'][flagchan] = np.nan 
    np.save("_tempdsI.npy", data['dsI'])
    outputs['dsI_1'] = data['dsI'].copy()
    outputs['time_1'], outputs['freq_1'] = data['time'].copy(), data['freq'].copy()

    # first run of find_frb
    outputs['tcrop_1'], outputs['fcrop_1'], tcrop_frame, outputs['tI_1'], outputs['fI_1'], outputs['smth_1'] = find_frb(
                                                    args = args, i = "_tempdsI.npy", toffset = outputs['windowcrop'][0],
                                                    ifcrop = outputs['fcrop_0'])


    # RFI subtraction and coarse RFI flagging
    frb.set(t_crop = outputs['tcrop_1'])
    rfisubs, outputs['rfisub_points'] = get_rfisubtractions(args, frb)
    outputs['rfisubs'] = rfisubs.copy()

    # only do flagging on off-pulse region
    dsI_lhs = frb.get_data("dsI", tN = args.tN, t_crop = [outputs['windowcrop'][0], outputs['tcrop_1'][0]], get = True)['dsI']
    dsI_rhs = frb.get_data("dsI", tN = args.tN, t_crop = [outputs['tcrop_1'][1], outputs['windowcrop'][1]], get = True)['dsI']
    dsI_off = np.concatenate((dsI_lhs, dsI_rhs), axis = 1)

    if args.rfisub:
        flagchan, outputs['stdfI_2'] = medrms_chanflag(dsI_off - rfisubs, args.thresh, 1, args.rfiter)
    else:
        flagchan = outputs['flagchan_1'].copy()
        outputs['stdfI_2'] = outputs['stdfI_1'].copy()

    outputs['flagchan_2'] = flagchan.copy()
    # outputs['flagchan_2'] = flagchan.copy()

    # get data for second run of find_frb
    data = frb.get_data("dsI", tN = 1, t_crop = outputs['windowcrop'], get = True)
    data['dsI'] -= rfisubs
    data['dsI'][flagchan] = np.nan
    np.save("_tempdsI.npy", data['dsI'])
    outputs['dsI_2'] = data['dsI'].copy()
    outputs['time_2'], outputs['freq_2'] = data['time'].copy(), data['freq'].copy()

    # second run of find_frb
    outputs['tcrop_2'], outputs['fcrop_2'], _, outputs['tI_2'], outputs['fI_2'], outputs['smth_2'] = find_frb(args = args, i = "_tempdsI.npy",
                                                toffset = outputs['windowcrop'][0], ifcrop = outputs['fcrop_1'])

    # remove any temporay files
    os.remove("_tempdsI.npy")

    print(outputs['flagchan_2'])

    return outputs











def medrms_chanflag(dsI, threshold = 3.0, tN = 1000, iter = 1):
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
    iter : int
        Number of iterations

    Returns
    -------
    flagchans : 1D np.ndarray
        Array of channel indicies where RFI was detected
    
    """

    flagchan = np.array([], dtype = int)

    # std and median calculations
    stdfI = np.nanstd(average(dsI, axis = 1, N = tN), axis = 1)
    stdfIcopy = stdfI.copy()
    for i in range(iter):
        stdfIabs = np.abs(stdfI - np.nanmedian(stdfI))
        flagchan_i = np.where(stdfIabs > (threshold * np.nanmedian(stdfIabs)))[0]
        stdfI[flagchan_i] = np.nan 
        flagchan = np.concatenate((flagchan, flagchan_i))

    return flagchan, np.abs(stdfIcopy - np.nanmedian(stdfIcopy))






def subband_search(args, outputs, dsI):
    """
    """

    # split into N subbands
    nchan = dsI.shape[0]
    Nsubchan = nchan // args.Nband 
    subband_nchans = [Nsubchan] * args.Nband 
    dt = args.dt * args.tN 

    j = 0
    for i in range(nchan - Nsubchan * args.Nband):
        subband_nchans[j] += 1
        j += 1
        if j == args.Nband - 1:
            j = 0
        
    # now all channels are put into subbands
    print("Subband Nchans:")
    print(subband_nchans)

    # split these subbands and find maximum peak
    tIsub = []
    j = 0
    peaksamp_sub = np.zeros(args.Nband, dtype = int)
    peakval_sub = np.zeros(args.Nband, dtype = float)
    peakpos_sub = np.zeros(args.Nband, dtype = float)
    for i in range(args.Nband):
        tIsub += [np.nansum(dsI[j:j+subband_nchans[i]], axis = 0) / subband_nchans[i]]
        j += subband_nchans[i]

        # get peak
        peaksamp_sub[i] = np.argmax(tIsub[i])
        peakval_sub[i] = tIsub[i][peaksamp_sub[i]]
        peakpos_sub[i] = peaksamp_sub[i] * dt
    
    # get maximum subband, then check which subbands maximum point are within said bounds using args.w
    peaksamp = peaksamp_sub[np.argmax(peakval_sub)]
    peakpos = peaksamp * dt

    tcrop = [peakpos - args.w/2, peakpos + args.w/2]

    signal_bool = np.zeros(args.Nband, dtype = bool)
    for i in range(args.Nband):
        if (peakpos_sub[i] > tcrop[0]) and (peakpos_sub[i] < tcrop[1]):
            signal_bool[i] = True
    
    bandind = np.arange(args.Nband)[signal_bool]
    topsubband, bottomsubband = np.min(bandind), np.max(bandind)

    # return fcrop
    df = args.bw / nchan
    freqs = np.linspace(args.cfreq + args.bw/2 - df/2, 
                        args.cfreq - args.bw/2 + df/2, nchan)
    
    subband_nchans = np.array(subband_nchans)
    if topsubband == 0:
        fmax = freqs[0]
    else:
        fmax = freqs[np.sum(subband_nchans[:topsubband])]
    
    if bottomsubband == 0:
        fmin = freqs[subband_nchans[0]]
    else:
        fmin = freqs[np.sum(subband_nchans[:bottomsubband+1])-1]

    # set outputs
    outputs['subband_peaksamps'] = peaksamp_sub.copy()
    outputs['subband_peakpos'] = peakpos_sub.copy()
    outputs['subband_tIs'] = tIsub.copy()
    outputs['subband_nchans'] = subband_nchans.copy()
    outputs['subband_topband'] = topsubband
    outputs['subband_bottomband'] = bottomsubband
    outputs['subband_bool'] = signal_bool.copy()
    print([fmin, fmax])

    return peaksamp, [fmin, fmax]








def center_fluence(args, frb, flagchan, windowcrop, iter = 2):
    """
    Center windowcrop to the median of the fluence, i.e. 50% fluence on either side
    
    """

    for i in range(iter):
        dsI = frb.get_data("dsI", t_crop = windowcrop, get = True)['dsI']
        tI = np.nanmean(dsI[~flagchan], axis = 0)
        fluence = np.sum(tI)

        tIcumsum = np.cumsum(tI)
        centroid = np.argmin(np.abs(tIcumsum - fluence/2))
        sampoffset = centroid - tI.size // 2

        # update window crop
        windowcrop[0] += frb.par.dt * sampoffset
        windowcrop[1] += frb.par.dt * sampoffset

    centroid = windowcrop[0] + 0.5 * (windowcrop[1] - windowcrop[0])
    
    return centroid, windowcrop












def find_frb(args, i, toffset = 0.0, ifcrop = ['min', 'max']):
    """
    Find frb
    
    Parameters
    ----------
    i : str
        filename for temporary dsI.npy file
    toffset : float
        Offset to apply to tcrop before returning
    ifcrop : list[float]
        initial fcrop to apply when scrunching in freq to get time series data

    """

    # load data
    frb = FRB(cfreq = args.cfreq, bw = args.bw, dt = args.dt)
    frb.load_data(dsI = i)
    frb.set(df = args.bw/frb.par.nchan)

    # find time crop
    tI = frb.get_data("tI", f_crop = ifcrop, get = True)['tI']
    centroid, lw, rw = find_optimal_fluence_width(tI = tI, yfrac = args.yfrac, mode = "min")
    tcrop = [(centroid - lw) * args.dt, (centroid + rw) * args.dt]

    # weight time data
    wavg = 1.0  # average of weights
    tW = False
    tIsmth = None
    if args.stDev > 0:
        tIsmth = frb.get_data("tI", f_crop = ifcrop, t_crop = tcrop, get = True)['tI']
        gaussian_W = gaussian_smooth(tIsmth, args.stDev)
        tIsmth = np.interp(np.linspace(0, 1.0, tIsmth.size),
                np.linspace(0, 1.0, gaussian_W.size), gaussian_W)
        wavg = np.sum(tIsmth)
        tW = True
    frb.par.set_weights(xtype = "t", method = "None", W = tIsmth)
    frb.set(apply_tW = tW)

    # find freq crop
    if args.fcrop_method == "minfluence":
        fI = frb.get_data("fI", t_crop = tcrop, get = True)['fI']
        fIzeroed = fI.copy()
        fIzeroed[np.isnan(fIzeroed)] = 0.0
        centroid, lw, rw = find_optimal_fluence_width(tI = fIzeroed, yfrac = args.yfrac, mode = "min")
        fcrop = [frb.par.f_lim[1] - (centroid + rw) * frb.par.df, 
                 frb.par.f_lim[1] - (centroid - lw) * frb.par.df]
        
    elif args.fcrop_method == "fsig":
        terrcrop = [tcrop[0] - args.rmsw, tcrop[0]]
        if terrcrop[0] < frb.par.t_lim[0]:
            print("[terr crop] Too few samples on LHS, attempting RHS...")
            terrcrop = [tcrop[1], tcrop[1] + args.rmsw]
            if terrcrop[1] > frb.par.t_lim[1]:
                print("[terr crop] Too few samples on RHS, aborting...")
                ValueError("Try reducing --rmsw!")

        fdata = frb.get_data("fI", t_crop = tcrop, terr_crop = terrcrop, get = True)

        # filter out channels lower than --fsig
        mask = fdata['fI'] < args.fsig * fdata['fIerr'] * wavg
        chans = np.ones(fdata['fI'].size, dtype = float)
        chans[mask] = np.nan 
        chan_idx = np.where(~np.isnan(chans))[0]
        fcrop = [fdata['freq'][np.max(chan_idx)], fdata['freq'][np.min(chan_idx)]]
        fI = fdata['fI']

    return [tcrop[0] + toffset, tcrop[1] + toffset], fcrop, tcrop, tI, fI, tIsmth


        






def get_rfisubtractions(args, frb, stk = "I"):
    """
    get rfi subtractions
    
    """

    if not args.rfisub:
        return np.zeros(frb.par.nchan, dtype = float)[:, None], []

    dsI = frb.get_data(f'ds{stk}', get = True)[f'ds{stk}']

    lhsflag = True 
    rhsflag = True

    # make windows
    lrficrop = [frb.metapar.t_crop[0] - args.rmsw, frb.metapar.t_crop[0] - args.rmsw/2]
    if lrficrop[1] < frb.par.t_lim[0]:
        lhsflag = False
        ValueError("[RFI subtraction] Burst too close to LHS boundary, cant do rfi subtraction!")
    if lrficrop[0] < frb.par.t_lim[0]:
        print("[RFI subtraction] cropping LHS rfi window...")
        lrficrop[0] = frb.par.t_lim[0]

    rrficrop = [frb.metapar.t_crop[1] + args.rmsw/2, frb.metapar.t_crop[1] + args.rmsw]
    if rrficrop[0] > frb.par.t_lim[1]:
        rhsflag = False
        ValueError("[RFI subtraction] Burst too close to RHS boundary, cant do rfi subtraction!")
    if rrficrop[1] > frb.par.t_lim[1]:
        print("[RFI subtraction] cropping RHS rfi window...")
        rrficrop[1] = frb.par.t_lim[1]

    # get rfi data
    lrfi, rrfi = np.array([], dtype = float), np.array([], dtype = float) 
    rfisub_points = []
    if lhsflag:
        rfisub_points += lrficrop
        lrfi = frb.get_data(f'ds{stk}', t_crop = lrficrop, get = True)[f'ds{stk}']
        if not rhsflag:
            rfi = lrfi
    if rhsflag:
        rfisub_points += rrficrop
        rrfi = frb.get_data(f'ds{stk}', t_crop = rrficrop, get = True)[f'ds{stk}']
        if not lhsflag:
            rfi = rrfi

    if (not lhsflag) and (not rhsflag):
        print("No free off-pulse region to get RFI subtraction! Skipping step!")
        return np.zeros(frb.par.nchan, dtype = float)[:, None], []

    if lhsflag and rhsflag:
        rfi = np.concatenate((lrfi, rrfi), axis = 1)

    return np.mean(rfi, axis = 1)[:, None], rfisub_points













def plot_diagnostics(args, outputs):
    """
    Make various plots on the processing and outputs of the findfrb.py script
    
    """

    ## PLOT DYNSPECS BEFORE AND AFTER ##
    ## AT VARIOUS STAGES              ##

    ## plot dynspecs
    plot_ds_flags(args, outputs)    
    
    ## PLOT FRB CROPPING AT STAGE 2 ##
    plot_ds_crop(args, outputs, stage = 2)

    if args.v:
        # plot peak finding
        plot_peak_finding(args, outputs)    

        ## PLOT FRB CROPPING AT STAGE 1 ##
        plot_ds_crop(args, outputs, stage = 1)

        ## PLOT RFI flagging results ##
        plot_rfi_flagging(args, outputs)

        ## plot subband results ##
        if args.Nband > 1:
            plot_subbands(args, outputs)

        # ## testing
        # model_cumsum(args, outputs)
        # plot_fluence_vs_timeres(args, outputs)

    if args.p:
        plt.show()







def plot_ds_flags(args, outputs):
    """"""

    def set_nanchans_to_zero(dsI):
        dsI[np.isnan(dsI[:, 0])] = 0.0

    def plot_rectangle(ax, tcrop, fcrop, color = 'r'):
        ax.add_patch(Rectangle((tcrop[0], fcrop[0]), 
            tcrop[-1] - tcrop[0], fcrop[-1] - fcrop[0], facecolor = 'None',
            edgecolor = 'r', ls = '--', alpha = 0.7, lw = 0.5))

    def plot_rfisub_points(ax):
        ylim = ax.get_ylim()

        for i in range(len(outputs['rfisub_points'])//2):
            plot_rectangle(ax, outputs['rfisub_points'][0 + 2*i:2 + 2*i], 
                            outputs['fcrop_2'], color = 'linegreen')
        
        ax.set_ylim(ylim)

    # get ds at tN for args.pw
    frb = FRB(cfreq = args.cfreq, bw = args.bw, dt = args.dt)
    frb.load_data(dsI = args.i)
    frb.set(df = args.bw / frb.par.nchan)
    
    tcrop = [outputs['peakpos'] - args.pw/2, outputs['peakpos'] + args.pw/2]
    if tcrop[0] < frb.par.t_lim[0]:
        tcrop[0] = frb.par.t_lim[0]
    if tcrop[1] > frb.par.t_lim[1]:
        tcrop[1] = frb.par.t_lim[1]

    print(tcrop)
    
    data = frb.get_data("dsI", t_crop = tcrop, tN = args.tN, get = True)

    print(data['dsI'])
    print(data['dsI'].shape)

    # create figure
    fig, ax = plt.subplots(1, 3, figsize = (14, 8))
    ax = ax.flatten()

    # plot unaltered figure
    extent = [*frb.this_par.t_lim, *frb.this_par.f_lim]
    set_nanchans_to_zero(data['dsI'])
    ax[0].imshow(data['dsI'], aspect = 'auto', extent = extent, 
                    interpolation = "None")
    plot_rectangle(ax[0], outputs['tcrop_2'], outputs['fcrop_2'])
    plot_rfisub_points(ax[0])
    ax[0].set_ylabel("Freq [MHz]", fontsize = 16)                
    ax[0].set_xlabel("Time [ms]", fontsize = 16)
    ax[0].set_title("Raw")

    # plot flagchan stage 1
    dsIflagged = data['dsI'].copy()
    dsIflagged[outputs['flagchan_1']] = 0.0
    ax[1].imshow(average(dsIflagged, N = args.pfN, axis = 0), aspect = 'auto', extent = extent, 
                interpolation = "None")
    plot_rectangle(ax[1], outputs['tcrop_2'], outputs['fcrop_2'])
    plot_rfisub_points(ax[1])
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_xlabel("Time [ms]", fontsize = 16)
    ax[1].set_title("Coarse flagging")

    # plot flagchan stage 2
    dsIflagged = data['dsI'].copy()
    dsIflagged -= outputs['rfisubs']
    dsIflagged[outputs['flagchan_2']] = 0.0
    ax[2].imshow(average(dsIflagged, N = args.pfN, axis = 0), aspect = 'auto', extent = extent,
                interpolation = "None")
    plot_rectangle(ax[2], outputs['tcrop_2'], outputs['fcrop_2'])
    plot_rfisub_points(ax[2])
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_xlabel("Time [ms]", fontsize = 16)
    rfistr = "(No rfi subtractions)"
    if args.rfisub:
        rfistr = "(With rfi subtractions)"
    ax[2].set_title("Coarse flagging " + rfistr)

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(args.o + "_RFIdynspecs.png")








def plot_peak_finding(args, outputs):

    fig, ax = plt.subplots(2, 1, figsize = (8, 12), sharex = True, gridspec_kw = {'height_ratios':[1, 3]})
    ax = ax.flatten()

    outputs['dsI_full'][outputs['flagchan_1']] = 0.0
    outputs['dsI_full'][np.isnan(outputs['dsI_full'][:,0])] = 0.0
    ax[1].imshow(outputs['dsI_full'], aspect = 'auto', extent = [*outputs['full_xextent'],
                    args.cfreq - args.bw/2, args.cfreq + args.bw/2], interpolation = "None")
    ax[1].set_ylabel("Freq [MHz]", fontsize = 16)
    ax[1].set_xlabel("Time [ms]", fontsize = 16)
    
    tI = np.nanmean(outputs['dsI_full'], axis = 0)
    ax[0].plot(np.linspace(*outputs['full_xextent'], tI.size), tI, 'k')
    ax[0].set_ylabel("Flux Density (arb.)", fontsize = 16)
    ax[0].get_xaxis().set_visible(False)
    ylim = ax[0].get_ylim()
    ax[0].plot([outputs['peakpos']]*2, ylim, 'r--')
    ax[0].set_ylim(ylim)

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(args.o + "_peak.png")










def plot_ds_crop(args, outputs, stage = 1):
    """"""

    def ax_remove_labels(ax):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def set_nanchans_to_zero(dsI):
        dsI[np.isnan(dsI[:, 0])] = 0.0
    
    fig, ax = plt.subplots(3, 3, figsize = (12.5, 12.5), gridspec_kw = {'width_ratios': [6, 2, 1], 'height_ratios':[1, 2, 6]})
    ax = ax.flatten()

    for i in [1, 2, 4, 5]:
        ax[i].set_axis_off()

    ax[3].sharex(ax[6])
    ax[0].sharex(ax[6])
    ax[7].sharey(ax[6])
    ax[8].sharey(ax[6])

    # plot dynamic spectrum
    t = outputs[f'time_{stage}']
    f = outputs[f"freq_{stage}"]
    set_nanchans_to_zero(outputs[f'dsI_{stage}'])
    ax[6].imshow(average(outputs[f'dsI_{stage}'], axis = 1, N = args.tN), aspect = 'auto', interpolation = "None", 
                 extent = [t[0], t[-1], f[-1], f[0]])
    ax[6].add_patch(Rectangle((outputs[f'tcrop_{stage}'][0], outputs[f'fcrop_{stage}'][0]), 
                    outputs[f'tcrop_{stage}'][-1] - outputs[f'tcrop_{stage}'][0],
                    outputs[f'fcrop_{stage}'][-1] - outputs[f'fcrop_{stage}'][0], facecolor = 'None',
                    edgecolor = 'r', ls = '--', alpha = 0.7, lw = 0.7))
    plot_channel_flagging_patch(ax[6], outputs[f'flagchan_{stage}'], f.size)
    ax[6].set_xlabel("Time [ms]", fontsize = 16)
    ax[6].set_ylabel("Freq [MHz]", fontsize = 16)

    # time series
    ax[3].plot(t, outputs[f'tI_{stage}'], 'k')
    ylim = ax[3].get_ylim()
    ax[3].plot([outputs[f'tcrop_{stage}'][0]]*2, ylim, 'r--')
    ax[3].plot([outputs[f'tcrop_{stage}'][1]]*2, ylim, 'r--')
    ax[3].set_ylim(ylim)
    ax_remove_labels(ax[3])

    # plot smoothed data
    if args.stDev > 0:
        ax[3].plot(np.linspace(*outputs[f'tcrop_{stage}'], outputs[f'smth_{stage}'].size), 
                    outputs[f'smth_{stage}'], 'r', label = "smoothed data")
        ax[3].legend()

    # freq spectra
    ax[7].plot(outputs[f'fI_{stage}'], f, 'k')
    xlim = ax[7].get_xlim()
    ax[7].plot(xlim, [outputs[f'fcrop_{stage}'][0]]*2, 'r--')
    ax[7].plot(xlim, [outputs[f'fcrop_{stage}'][1]]*2, 'r--')
    ax[7].set_xlim(xlim)
    ax_remove_labels(ax[7])

    # plot cumalitive time sum
    ax[0].plot(t, np.cumsum(outputs[f'tI_{stage}']), 'k')
    ax_remove_labels(ax[0])

    # plot cumalitive freq sum
    fIzeroed = outputs[f"fI_{stage}"].copy()
    fIzeroed[np.isnan(fIzeroed)] = 0.0
    ax[8].plot(np.cumsum(fIzeroed), f, 'k')
    ax[8].set_ylim([outputs[f'freq_{stage}'][-1], outputs[f'freq_{stage}'][0]])
    ax_remove_labels(ax[8])

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(args.o + f"_frbcrop_stage{stage}.png")

    return








def plot_rfi_flagging(args, outputs):

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    ax = ax.flatten()
    
    thresh1 = args.thresh * np.nanmedian(outputs['stdfI_1'])
    thresh2 = args.thresh * np.nanmedian(outputs['stdfI_2'])

    # plot first and second iteration
    ax[0].set_title("Stage 1")
    ax[0].plot(outputs['freq_1'], outputs['stdfI_1'], alpha = 0.8, linewidth = 2.0, 
            label = "No RFI flagging")
    xlim = ax[0].get_xlim()
    ax[0].set_xlim(xlim)
    ax[0].plot(xlim, [thresh1]*2, 'r--', label = "Threshold [--args.thresh]")
    ax[0].get_xaxis().set_visible(False)
    ax[0].legend()

    
    stdfI_1flagged = outputs['stdfI_1'].copy()
    stdfI_1flagged[outputs['flagchan_1']] = np.nan
    ax[2].plot(outputs['freq_1'], stdfI_1flagged, alpha = 0.8, linewidth = 2.0, 
            label = "RFI flagging")
    ax[2].plot(xlim, [thresh1]*2, 'r--')
    ax[2].sharex(ax[0])
    ax[2].legend()
    ax[2].set_xlabel("Freq [MHz]", fontsize = 16)


    stg2label = "No RFI flagging"
    if args.rfisub:
        stg2label = "RFI subtraction"

    ax[1].set_title("Stage 2")
    ax[1].plot(outputs['freq_2'], outputs['stdfI_2'], alpha = 0.8, linewidth = 2.0,
            label = stg2label)
    xlim = ax[1].get_xlim()
    ax[1].set_xlim(xlim)
    ax[1].plot(xlim, [thresh2]*2, 'r--')
    ax[1].get_xaxis().set_visible(False)

    stdfI_2flagged = outputs['stdfI_2'].copy()
    stdfI_2flagged[outputs['flagchan_2']] = np.nan 
    ax[3].plot(outputs['freq_2'], stdfI_2flagged, alpha = 0.8, linewidth = 2.0,
            label = "RFI flagging")
    ax[3].plot(xlim, [thresh2]*2, 'r--')
    ax[3].sharex(ax[1])
    ax[3].legend()
    ax[3].set_xlabel("Freq [MHz]", fontsize = 16)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0)

    plt.savefig(args.o + "_RFIflagging.png")
    










def plot_channel_flagging_patch(ax, flagchan, nchan):
    """
    Show flagged channels as bright patch on the side of dynamic spectrum

    Parameters
    ----------
    ax : Axes
        Axes object 
    flagchan : 1D np.ndarray or array-like
        flagged channel indicies 
    nchan : int
        Number of channels 
    
    """
    
    chans = np.ones(nchan, dtype = float) * np.nan
    chans[flagchan] = 0.55

    # plot patch
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xwidth = xlim[1] - xlim[0]
    ax.imshow(chans.reshape(chans.size, 1), aspect = 'auto', cmap = 'OrRd',
                vmax = 1, vmin = 0, extent = [xlim[0], xlim[0] + 0.02*xwidth, *ylim])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return








def model_cumsum(args, outputs):

    def sigmoid(x, a, b, c, d):
        return d*np.exp(a*(x - b)) / (1 + np.exp(a*(x - b))) + c


    fig, ax = plt.subplots(1, 1, figsize = (10, 6))
    x = outputs['time_2']
    y = np.cumsum(outputs['tI_2'])

    p, _ = curve_fit(sigmoid, x, y, p0 = [1.0, outputs['time_2'][outputs['time_2'].size//2], np.mean(y), 1.0])

    ax.scatter(x, y, c = 'k', marker = '+')
    ax.plot(x, sigmoid(x, *p), 'r', label = "Sigmoid model")

    ax.set_title("Cumsum of Burst fluence")
    ax.set_xlabel("Time [ms]", fontsize = 16)
    ax.set_ylabel("np.cumsum(tI)")

    fig.tight_layout()




def plot_fluence_vs_timeres(args, outputs):

    fig, ax = plt.subplots(1, 1, figsize = (10, 6))

    for i in [1, 5, 10, 20, 50]:
        tI = average(outputs['tI_2'], N = i)
        cumsum = np.cumsum(tI)
        x = np.linspace(outputs['time_2'][0], outputs['time_2'][-1], tI.size)
        ax.plot(x, cumsum, label = str(i), alpha = 0.7)
    
    ax.legend()

    fig.tight_layout()






def plot_subbands(args, outputs):

    fig, ax = plt.subplots(1, 1, figsize = (8, 14))

    yoffset = np.max(outputs['subband_tIs'][0])
    for i in range(args.Nband):
        ax.plot(outputs['subband_tIs'][i], label = f"subband {i}", alpha = 0.5)

    print(outputs["subband_peaksamps"])
    print(outputs['fcrop_0'])
    print(outputs['subband_peakpos'])

    ax.legend()




def save_outputs(args, outputs):
    """
    Save output parameters and crops
    
    """
    if args.parfile is not None:
        stkfiles = ilexIO(filepath = args.parfile).load_pars()['data']
    frb = FRB(cfreq = args.cfreq, bw = args.bw, dt = args.dt)
    frb.load_data(**stkfiles)

    tcrop = outputs['tcrop_2'].copy()
    tcrop = [tcrop[0] - args.cropds/2, tcrop[1] + args.cropds/2]
    if tcrop[0] < frb.par.t_lim[0]:
        tcrop[0] = frb.par.t_lim[0]
    if tcrop[1] > frb.par.t_lim[1]:
        tcrop[1] = frb.par.t_lim[1]

    # write outputs to txt file
    with open(args.o + "_summary.txt", 'w') as file:
        file.write("Final results:\n")
        file.write(f"burst tcrop: {outputs['tcrop_2']}\n")
        file.write(f"burst fcrop: {outputs['fcrop_2']}\n")
        file.write(f"output tcrop: {tcrop}\n")
        rfisubflag = "FALSE"
        if args.rfisub:
            rfisubflag = "TRUE"
        file.write("RFI subtraction: " + rfisubflag + "\n")
        file.write("flagged channels: ")
        for chan in outputs['flagchan_2']:
            file.write(f"{chan} ")
        file.write("\n")

        chans = np.ones(outputs['freq_2'].size)
        chans[outputs['flagchan_2']] = np.nan
        zapchan = get_zapstr(chans, outputs['freq_2'])
        file.write("flagged frequencies: ")
        file.write(zapchan)
        file.write("\n")

    return 








def save_data(args, outputs):

    if args.parfile is not None:
        stkfiles = ilexIO(filepath = args.parfile).load_pars()['data']
    frb = FRB(cfreq = args.cfreq, bw = args.bw, dt = args.dt)
    print(stkfiles)
    frb.load_data(**stkfiles)
    frb.set(df = args.bw / frb.par.nchan)

    print(frb)

    # save stokes I dynspec
    stkfiles = frb._data_files
    if args.save_data:

        # crop ds if applicable
        tcrop = frb.par.t_lim.copy()
        if args.cropds is not None:
            tcrop = outputs['tcrop_2'].copy()
            tcrop = [tcrop[0] - args.cropds/2, tcrop[1] + args.cropds/2]
            if tcrop[0] < frb.par.t_lim[0]:
                tcrop[0] = frb.par.t_lim[0]
            if tcrop[1] > frb.par.t_lim[1]:
                tcrop[1] = frb.par.t_lim[1]
            print(f"After taking into account the bounds of the ds buffer, the total width of saved dataset is {tcrop[1] - tcrop[0]:.2f} ms")

        stkfiles2save = stkfiles.copy()
        for S in "IQUV":
            if stkfiles[f"ds{S}"] is not None:
                print(f"Processing and saving Stokes {S} Dynamic Spectrum")
                frb.set(t_crop = outputs['tcrop_1'])
                rfisubs, _ = get_rfisubtractions(args, frb, stk = S)
                ds = frb.get_data(f"ds{S}", t_crop = tcrop, get = True)[f'ds{S}']
                ds -= rfisubs

                ds[outputs['flagchan_2']] = np.nan 

                np.save(args.o + f"_ds{S}.npy", ds)
                stkfiles2save[f"ds{S}"] = args.o + f"_ds{S}.npy"

        outputs['tcrop_2'][0] -= (tcrop[0] - frb.par.t_lim[0])
        outputs['tcrop_2'][1] -= (tcrop[0] - frb.par.t_lim[0])
        frb.load_data(dsI = stkfiles2save['dsI'], dsQ = stkfiles2save['dsQ'],
                      dsU = stkfiles2save['dsU'], dsV = stkfiles2save['dsV'])

    overwrite = False
    if args.parfile is not None:
        if args.oparfile == args.parfile:
            overwrite = True
    frb.set(t_crop = outputs['tcrop_2'], f_crop = outputs['fcrop_2'])
    ilexio = ilexIO(filepath = args.oparfile, frb = frb, overwrite = overwrite, datafilepath = args.o)
    ilexio.save()

    return



        






if __name__ == "__main__":

    # args
    args = get_args()
    load_data(args)
    print(args)

    # locate frb
    outputs = main(args)
    
    # Save summary of findfrb run
    save_outputs(args, outputs)

    # plot stuff
    plot_diagnostics(args, outputs)

    # save data
    save_data(args, outputs)

    print(outputs['flagchan_2'])

    print("[findfrb.py] completed!")