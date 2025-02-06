## imports
from .data import *
from .logging import log, get_verbose
from .utils import get_stk_from_datalist
from .globals import _G
from time import time
import matplotlib.pyplot as plt


##==========================##
## MAIN PROCESSING FUNCTION ##
##==========================##

def master_proc_data(stk, freq, base_data_list, par: dict = {}, debias = False, ratio = False,
                        ratio_rms_threshold = None):
    """
    Info:
        Master function for processing data


    Parameters
    ----------
    stk: dict[np.memmap]
        Dictionary of memory maps of Stokes Dynamic spectra
    freq: (ndarray) 
        Array of frequencies before cropping/averaging (processing)
    base_data_list: List 
        List of requested data products
    par: Dict 
        Parameters for data processing, the following are required \n
        [t_crop]: Crop in Time phase (default = [0.0, 1.0]) \n
        [f_crop]: Crop in Freq phase (default = [0.0, 1.0]) \n
        [terr_crop]: Crop of Error Time phase (default = None) \n
        [tN]: Factor in Time average (default = 1) \n
        [fN]: Factor in Freq average (default = 1) \n
        [tW]: Time weights (default = None) \n
        [fW]: Freq weifhts (default = None) \n
        [norm]: Method of normalisation (defualt = "None") \n
        [RM]: Rotation Measure (default = 0.0) \n
        [cfreq]: Central Frequency (default = 1271.5) \n
        [f0]: reference frequency (default = cfreq) \n
        [pa0]: PA at f0 (default = 0.0)
    debias: bool, optional
        if True, debias L and P polarisations if requested
    ratio: bool, optional
        If True, calculates Stokes ratios X/I for all t and f products
    ratio_rms_threshold: float, optional
        Mask Stokes ratios by ratio_rms_threshold * rms

    Returns
    -------
    _ds : dict
        Dictionary of processed dynamic spectra
    _t :dict 
        Dictionary of processed time series data
    _f : dict
         Dictionary of processed spectra data 
    freq : ndarray
         Array of Frequencies in MHz
    
    """
    ## ============================== ##
    ##       SET UP CONTAINERS        ##
    ## ============================== ##

    stk_data = {}
    for key in stk.keys():
        stk_data[key] = stk[key]
    stk_data['freq'] = freq.copy()

    par = _proc_par(par)    
    
    data_list = base_data_list.copy()


    # flags
    err_flag = False
    if par['terr_crop'] is not None:
        err_flag = True

    zap_flag = False
    if (par['zapchan'] is not None) and (par['zapchan'] != ""):
        zap_flag = True

    fw_flag = par['fW'] is not None
    tw_flag = par['tW'] is not None
    

    if not err_flag:
        debias = False
        ratio_rms_threshold = None

    # update data_list in case pre-requisists are needed
    if "tL" in data_list:
        data_list += ["tQ", "tU"]
    if "fL" in data_list:
        data_list += ["fQ", "fU"]
    if "tP" in data_list:
        data_list += ["tQ", "tU", "tV"]
    if "fP" in data_list:
        data_list += ["fQ", "fU", "fV"]
    if debias or ratio:
        data_list += ["tI", "fI"]
    
    data_list = list(set(data_list))    
    
    # Create empty data containers
    _ds, _t, _f, stk_list, ds_list, t_list, f_list = _get_containers_and_lists(data_list) 

    fday_flag = False
    if "Q" in stk_list and "U" in stk_list:
        fday_flag = True 

    L_flag = ("tL" in data_list) or ("fL" in data_list)
    P_flag = ("tP" in data_list) or ("fP" in data_list)


    # get verbose
    verbose = get_verbose()

    # container to store temporary data in
    stk_ds = {}

    # additional diagnostics container, holds flags to log processing steps
    # [Timecrop, freqcrop, chanzap, normalising, Faraday derotating, Time averaging, Freq averaging, making products]
    tick = " "
    if err_flag:
        tick = "X"
    log(f"PROCESSING: Data [X], Error [{tick}]\n", lpf_col = "lgreen")
    _log_progress([True, True, zap_flag, True, fday_flag, True, True, tw_flag, fw_flag, L_flag, P_flag, debias, ratio], par,
                    ratio_rms_threshold)

    ## ============================== ##
    ##         RUN PROCESSING         ##
    ## ============================== ##
    # log("Processing data", lpf = False)


    ## crop ##    
    log(f"Applying On-Pulse Time Crop = {par['t_crop']}", lpf = False)
    log(f"Applying Off-Pulse Time crop = {par['terr_crop']}", lpf = False)
    log(f"Applying Freq Crop = {par['f_crop']}", lpf = False)
    stk_ds = _crop(stk = stk, stk_ds = stk_ds, stk_list = stk_list, 
                    par = par, err = err_flag)

    # also crop freq array
    freq = pslice(freq, *par['f_crop'])



    ## Channel Zap ##
    if zap_flag:
        log(f"Applying channel zapping", lpf = False)
        stk_ds, zap_flag = _zap_chan(stk = stk, stk_ds = stk_ds, stk_list = stk_list,
                                freq = freq, par = par, err = err_flag)
        
        if not zap_flag:
            log("All channel zapping lies outside the known bandwidth, will skip channel zapping", lpf = False, stype = "warn")


    ## normalise ##
    log(f"Applying normalisation = {par['norm']}", lpf = False)
    stk_ds = _normalise(stk = stk, stk_ds = stk_ds, stk_list = stk_list,
                        par = par, err = err_flag, zap = zap_flag)

    ## faraday rotate ##
    if fday_flag:
        log("Applying Faraday De-rotation", lpf = False)
        stk_ds = _faraday_rotate(stk = stk, stk_ds = stk_ds, stk_list = stk_list,
                                freq = freq, par = par, err = err_flag)

    
    # ## average ##
    log(f"Applying Time Averaging = {par['tN']}", lpf = False)
    log(f"Applying Freq Averaging = {par['fN']}", lpf = False)
    stk_ds = _average(stk = stk, stk_ds = stk_ds, stk_list = stk_list,
                    par = par, err = err_flag, zap = zap_flag)



    # also average freq array
    freq = average(freq, N = par['fN'], nan = zap_flag)


    ## ========================= ##
    ## SAVING DATA TO CONTAINERS ##
    ## ========================= ## 

    # diagnostic plotting
    if verbose:

        # plot frequency weights
        if "I" in stk_ds.keys():
            _diag_plot_weights(stk_ds['I'], par)


        # can add more later


    for S in stk_list:




        #------ dynamic spectra -------#
        if S in ds_list:
            _ds[S] = stk_ds[S].copy()






        #------- time series -------#
        if S in t_list:
            _t[S] = _scrunch(stk_ds[S], xtype = "t", W = _average_weight(par['fW'], par['fN']), nan = zap_flag)
            if err_flag:
                _t[f"{S}err"] = _time_err(stk_ds[f"{S}err"], nan = zap_flag)






        #----------- freq spectra -----------#
        if S in f_list:
            if err_flag:
                _f[f"{S}err"] = _freq_err(stk_ds[f"{S}err"], par['t_crop'], par['terr_crop'])
                
            _f[S] = _scrunch(stk_ds[S], xtype = "f", W = _average_weight(par['tW'], par['tN']))

        
        # del temp data
        del stk_ds[S]
        if err_flag:
            del stk_ds[f"{S}err"]

    
    #-------- L and P polarisations --------# 
    _retrieve_LP(_t, _f, data_list, debias = debias)


    # ----- Convert to Stokes ratios ----- #
    _stokes_ratios(_t, _f, ratio = ratio, ratio_rms_threshold = ratio_rms_threshold)

    

    # compile all flags for future use
    _flags = {'zap_flag':zap_flag, 'err_flag':err_flag}

        
    # return data containers
    return _ds, _t, _f, freq, _flags






def _average_weight(W, N):
    
    if W is None:
        return None
    
    if np.isscalar(W):
        return W
    
    if hasattr(W, "__len__"):
        return average(x = W, N = N)
    
    print("Someting went wrong with averaging weights!!")
    return None






# Helper functions for processing
def _proc_par(par):
    """
    Further process parameters
    """

    log("Checking Parameters for data processing", lpf_col="lgreen")

    # defaults
    def_m = deepcopy(_G.p)
    for key in ["tW", "fW"]:    # add extra params
        def_m[key] = None

    # check if any are missing
    keys = par.keys()
    parkeys = def_m.keys()
    metaparkeys = _G.mp.keys()

    for pk in parkeys:
        if pk not in keys:
            par[pk] = def_m[pk]
    for mpk in metaparkeys:
        if mpk not in keys:
            par[mpk] = _G.mp[mpk]
    
    # further case by case parameter processing

    # if f0 not specified, set to cfreq
    # if par['f0'] is None:
    #     log(f"[f0] unspecified, setting f0 = cfreq = {par['cfreq']}\n", lpf = False)
    #     par['f0'] = par['cfreq']
    
    return par




def _get_containers_and_lists(data_list):
    """
    Get empty containers and lits
    """

    # Create empty data containers
    _ds, _t, _f = {}, {}, {}

    for S in "IQUV":
        _ds[S], _ds[f"{S}err"] = None, None
    for S in "IQUVLP":
        _t[S], _t[f"{S}err"] = None, None
        _f[S], _f[f"{S}err"] = None, None
    
    _freq = None

    # seperate data_list 
    ds_list = []
    t_list = []
    f_list = []

    for data in data_list:
        if data[:2] == "ds":
            ds_list += [data]
        elif data[0] == "t":
            t_list += [data]
        elif data[0] == "f":
            f_list += [data]
    
    # Get Stokes for each data type
    stk_list = get_stk_from_datalist(data_list)
    ds_list = get_stk_from_datalist(ds_list)
    t_list = get_stk_from_datalist(t_list)
    f_list = get_stk_from_datalist(f_list)


    return _ds, _t, _f, stk_list, ds_list, t_list, f_list






def _crop(stk, stk_ds, stk_list, par, err = False):
    """
    Info:
        Full crop in Time and Frequency

    Args:
        See Master _proc
    """

    for S in stk_list:
        t1 = time()
        # crop on-burst
        stk_ds[S] = pslice(stk[S], *par['t_crop'], axis = 1)        # Time 
        stk_ds[S] = pslice(stk_ds[S], *par['f_crop'], axis = 0)     # Freq
        
        # off-burst
        if err:
            stk_ds[f"{S}err"] = pslice(stk[S], *par['terr_crop'], axis = 1)
            stk_ds[f"{S}err"] = pslice(stk_ds[f"{S}err"], *par['f_crop'], axis = 0)
    
    return stk_ds



def _scrunch(x, xtype = "t", W = None, nan = False):
    """
    scrunch data with either a box car or weights

    
    """

    if nan:
        func = np.nanmean
    else:
        func = np.mean


    if xtype == "t":    # if i'm making a time array, weight freqs
        return func(f_weight(x, W), axis = 0) 
    
    elif xtype == "f":  # if i'm making a freq array, weight times
        return func(t_weight(x, W), axis = 1)
    





def _weight(stk, stk_ds, stk_list, par, err = False):
    """
    Info:
        Weight in Time and Frequency

    Args:
        See master _proc
    """

    for S in stk_list:
        # weight on-pulse data
        stk_ds[S] = f_weight(stk_ds[S], par['fW'])        # Freq
        stk_ds[S] = t_weight(stk_ds[S], par['tW'])        # Time

        # off-burst
        if err:
            stk_ds[f"{S}err"] = f_weight(stk_ds[f"{S}err"], par['fW'])
    
    return stk_ds










def _normalise(stk, stk_ds, stk_list, par, err = False, zap = False):
    """
    Info:
        Do normalisation
    
    Args:
        see master _proc
        ex: normalisation constant
    """
    # ignore normalisation
    if par['norm'] == "None":
        return stk_ds

    # normalise by max of stokes I
    if par['norm'] == "maxI":
        nconst = np.max(stk['I'])

    for S in stk_list:
        # normalise by max of stokes S
        if norm == "max":
            if zap:
                nconst = np.nanmax(stk[S])
            else:
                nconst = np.max(stk[S])
            stk_ds[S] /= nconst
            stk_ds[f"{S}err"] /= nconst

        # normalise by abs max of stokes S
        elif norm == "absmax":
            if zap:
                nconst = np.nanmax(np.abs(stk[S]))
            else:
                nconst = np.max(np.abs(stk[S]))
            stk_ds[S] /= nconst
            stk_ds[f"{S}err"] /= nconst

        # normalise by max of stokes I
        elif norm == "maxI":
            stk_ds[S] /= nconst
            stk_ds[f"{S}err"] /= nconst

        else:
            log("Invalid method of normalisation, skipping step...", stype = "warn",
                lpf = False)
            return stk_ds
    
    return stk_ds






def _faraday_rotate(stk, stk_ds, stk_list, freq, par, err = False):
    """
    Info:
        Apply faraday de-rotation

    Args:
        see master _proc 
    """

    stk_ds['Q'], stk_ds['U'] = fday_rot(stk_ds['Q'], stk_ds['U'], f = freq, 
                                RM = par['RM'], f0 = par['f0'], pa0 = par['pa0'])

    return stk_ds






def _average(stk, stk_ds, stk_list, par, err = False, zap = False):
    """
    Info:
        Apply averaging in Time and Freq
    
    Args:
        see master _proc
    """

    for S in stk_list:
        ferr_flag = False
        if S in ["fIerr", "fQerr", "fUerr", "fVerr"]:   # freq errors need to be handled with care
            ferr_flag = True
        # average in time, don't want to average in time so as to
        # enable better std calculation
        if (par['tN'] > 1):
            stk_ds[S] = average(stk_ds[S], axis = 1, N = par['tN'])
            if err:
                stk_ds[f"{S}err"] = average(stk_ds[f"{S}err"], axis = 1, N = par['tN'])

        if par['fN'] > 1:
            stk_ds[S] = average(stk_ds[S], axis = 0, N = par['fN'], nan = zap)
            if err:
                stk_ds[f"{S}err"] = average(stk_ds[f"{S}err"], axis = 0, N = par['fN'], nan = zap)

    return stk_ds




def _time_err(ds, nan = False):
    """
    Info:
        Calculate error in Time
    
    Args:
        ds (ndarray): Dynamic spectra

    Returns:
        rms (float): average rms over time
    """

    if nan:
        func_mean = np.nanmean
        func_std = np.nanstd
    else:
        func_mean = np.mean
        func_std = np.std

    # scrunch in frequency
    t = func_mean(ds, axis = 0)

    # calculate rms
    rms = func_std(t)

    return rms



def _freq_err(ds, t_crop, terr_crop):
    """
    Info:
        Calculate error in Freq
    
    Args:
        ds (ndarray): Dynamic spectra

    Returns:
        rms (ndarray): per channel rms 
    """

    # calculate rms, also need to factor in the width of 
    # the error crop in time and inversely scale to that
    # this is assuming gaussian noise.

    crop_scale = ((terr_crop[1]-terr_crop[0])/(t_crop[1]-t_crop[0]))**0.5

    rms = crop_scale * np.std(ds, axis = 1) / ds.shape[1]**0.5

    return rms


# def _freq_err(ds):

#     rms = np.std(ds, axis = 1)
#     return rms





def _zap_chan(stk, stk_ds, stk_list, freq, par, err = False):
    """
    Zap channels, assumes contiguous frequency array
    
    """

    # vals
    df = freq[1] - freq[0]
    f_min = np.min(freq)
    f_max = np.max(freq)

    if df < 0:
        # upperside band
        fi = f_max
        df_step = -1

    else:
        # lowerside band
        fi = f_min
        df_step = 1

    df = abs(df)
    
    # split segments
    zap_segments = par['zapchan'].split(',')
    seg_idx = []

    # for each segment, check for delimiter :, else float cast
    for _, zap_seg in enumerate(zap_segments):

        # if segment is a range of frequencies
        if ":" in zap_seg:
            zap_range = zap_seg.strip().split(':')
            zap_0 = round(df_step * (float(zap_range[0]) - fi)/df)
            zap_1 = round(df_step * (float(zap_range[1]) - fi)/df)


            # check if completely outside bounds
            if (zap_0 < 0 and zap_1 < 0) or (zap_0 > freq.size -1 and zap_1 > freq.size -1):
                log(f"zap range [{zap_range[0]}, {zap_range[1]}] MHz out of range of bandwidth [{f_min}, {f_max}] MHz", lpf = False, stype = "warn")
                continue            
            
            # check bounds
            crop_zap = False

            if zap_0 < 0:
                crop_zap = True
                zap_0 = 0
            elif zap_0 > freq.size - 1:
                crop_zap = True
                zap_0 = freq.size - 1

            if zap_1 < 0:
                crop_zap = True
                zap_1 = 0
            elif zap_1 > freq.size - 1:
                crop_zap = True
                zap_1 = freq.size - 1

            if crop_zap:
                log(f"zap range cropped from [{zap_range[0]}, {zap_range[1]}] MHz -> [{freq[zap_0]}, {freq[zap_1]}] MHz", lpf = False, stype = "warn")

            seg_idx += list(range(zap_0,zap_1+df_step,df_step))[::df_step]
        
        # if segment is just a single frequency
        else:
            _idx = round(df_step * (float(zap_seg.strip()) - fi)/df)
            if (_idx < 0) or (_idx > freq.size - 1):
                log(f"zap channel {zap_seg.strip()} MHz out of bounds of bandwidth [{f_min}, {f_max}] MHz", lpf = False, stype = "warn")
            else:
                seg_idx += [_idx]


    # check if seg_idx is empty, i.e. if there is no channel zapping.
    if len(seg_idx) == 0:
        return stk_ds, False


    # mask out zapped channels
    for S in stk_list:
        stk_ds[S][seg_idx] = np.nan

        if err:
            stk_ds[f"{S}err"][seg_idx] = np.nan

    return stk_ds, True







def _retrieve_LP(_t, _f, data_list, debias = False):
    """
    Retrieve L and P data products if requested
    """

    # Linear polarisation
    if "tL" in data_list:
        if debias:
            _t["L"], _t["Lerr"] = calc_Ldebiased(_t['Q'], _t['U'], _t['Ierr'], 
                                                    _t['Qerr'], _t['Uerr'])
        else:
            _t["L"], _t["Lerr"] = calc_L(_t['Q'], _t['U'], _t['Qerr'], _t['Uerr'])

    if "fL" in data_list:
        if debias:
            _f["L"], _f["Lerr"] = calc_Ldebiased(_f['Q'], _f['U'], _f['Ierr'], 
                                                    _f['Qerr'], _f['Uerr'])
        else:
            _f["L"], _f["Lerr"] = calc_L(_f['Q'], _f['U'], _f['Qerr'], _f['Uerr'])

    # Total polarisation
    if "tP" in data_list:
        if debias:
            _t["P"], _t["Perr"] = calc_Pdebiased(_t['Q'], _t['U'], _t['V'], _t['Ierr'], 
                                                    _t['Qerr'], _t['Uerr'], _t['Verr'])
        else:
            _t["P"], _t["Perr"] = calc_P(_t['Q'], _t['U'], _t['V'], _t['Qerr'], _t['Uerr'],
                                         _t['Verr'])

    if "fP" in data_list:
        if debias:
            _f["P"], _f["Perr"] = calc_Pdebiased(_f['Q'], _f['U'], _f['V'], _f['Ierr'], 
                                                    _f['Qerr'], _f['Uerr'], _f['Verr'])
        else:
            _f["P"], _f["PLerr"] = calc_P(_f['Q'], _f['U'],_f['V'], _f['Qerr'], _f['Uerr'],
                                          _f['Verr'])


    return





def _stokes_ratios(_t, _f, ratio = False, ratio_rms_threshold = None):
    """
    Convert to Stokes ratios
    """

    if not ratio:
        return

    # create mask
    if ratio_rms_threshold is not None:
        t_mask = _t['I'] < ratio_rms_threshold * _t['Ierr']
        f_mask = _f['I'] < ratio_rms_threshold * _f['Ierr']

    # time series
    for s in "QUVLP":
        if _t[s] is not None:
            _t[s], _t[f"{s}err"] = calc_ratio(_t['I'], _t[s], _t['Ierr'], _t[f"{s}err"])
            if ratio_rms_threshold is not None:
                _t[s][t_mask] = np.nan
                _t[f"{s}err"][t_mask] = np.nan
    # freq series
    for s in "QUVLP":
        if _f[s] is not None:
            _f[s], _f[f"{s}err"] = calc_ratio(_f['I'], _f[s], _f['Ierr'], _f[f"{s}err"])
            if ratio_rms_threshold is not None:
                _f[s][f_mask] = np.nan
                _f[f"{s}err"][f_mask] = np.nan
    
    return













# diagnostics functions
def _log_progress(flags, par, ratio_rms_threshold = None):
    """
    Batch log progress
    
    """

    logs = [f"Time cropping [{par['t_crop'][0]}, {par['t_crop'][1]}]", f"Freq cropping [{par['f_crop'][0]}, {par['f_crop'][1]}]", 
            f"Channel zapping [{par['zapchan']}]", f"Normalising [{par['norm']}]", f"Faraday De-rotating [RM = {par['RM']}, f0 = {par['f0']}]", 
            f"Time averaging [N = {par['tN']}]", f"Freq averaging [N = {par['fN']}]", f"Time weighting", f"Freq weighting", 
            f"Calculating L (t and/or f)", f"Calculating P (t and/or P)", f"Debiasing L and/or P", f"Converting to Stokes ratios (t and f) [masking sigma = {ratio_rms_threshold}]"]

    pstr = ""

    for i, flag in enumerate(flags):
        tick = " "
        if flag:
            tick = "X"
        pstr += f"[{tick}]   " + logs[i] + "\n"

    
    log(pstr, lpf = False)




def _diag_plot_weights(ds, par):
    """
    Plot weights Against time array or freq array
    """

    # diagnostic plot for time weights
    if (par['tW'] is not None) and (ds is not None):
        plt.figure()
        t = np.nanmean(ds, axis = 0)
        weights = _average_weight(par['tW'], par['tN'])
        
        plt.plot(t, "--")
        plt.plot(weights/np.max(weights) * np.max(t))
        plt.xlabel("Time samples")
        plt.ylabel("arb.")
        
        # save fig
        filename = f"{par['name']}_applying_time_weights.png"
        log(f"Saving diagnostic plot of applying time weights: [{filename}]", lpf = False)
        plt.savefig(filename)

        plt.close()
    
    # diagnostic plot for freq weights
    if (par['fW'] is not None) and (ds is not None):
        plt.figure()
        f = np.mean(ds, axis = 1)
        weights = _average_weights(par['fW'], par['fN'])

        plt.plot(f, "--")
        plt.plot(weights/np.max(weights) * np.max(f))
        plt.xlabel("Freq channels")
        plt.ylabel("arb.")

        # save fig
        filename = f"{par['name']}_applying_freq_weights.png"
        log(f"Saving diagnostic plot of applying freq weights: [{filename}]", lpf = False)
        plt.savefig(filename)

        plt.close()
