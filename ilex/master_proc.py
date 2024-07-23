## imports
from .data import *
from .logging import log
from .utils import get_stk_from_datalist
from .globals import _G
from time import time

##==========================##
## MAIN PROCESSING FUNCTION ##
##==========================##

def master_proc_data(stk, freq, data_list, par: dict = {}):
    """
    Info:
        Master function for processing data


    Parameters
    ----------
    stk: dict[np.memmap]
        Dictionary of memory maps of Stokes Dynamic spectra
    freq: (ndarray) 
        Array of frequencies before cropping/averaging (processing)
    data_list: List 
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

    # Create empty data containers
    _ds, _t, _f, stk_list, ds_list, t_list, f_list = _get_containers_and_lists(data_list)

    # flags
    err_flag = False
    if par['terr_crop'] is not None:
        err_flag = True

    zap_flag = False
    if par['zapchan'] is not None:
        zap_flag = True

    fday_flag = False
    if "Q" in stk_list and "U" in stk_list:
        fday_flag = True 

    fw_flag = par['fW'] is not None
    tw_flag = par['tW'] is not None

    # container to store temporary data in
    stk_ds = {}


    ## ============================== ##
    ##         RUN PROCESSING         ##
    ## ============================== ##
    log("Processing ON-PULSE data", lpf = False)


    ## crop ##    
    log(f"Applying On-Pulse Time Crop = {par['t_crop']}", lpf = False)
    log(f"Applying Off-Pulse Time crop = {par['terr_crop']}", lpf = False)
    log(f"Applying Freq Crop = {par['f_crop']}", lpf = False)
    stk_ds = _crop(stk = stk, stk_ds = stk_ds, stk_list = stk_list, 
                    par = par, err = err_flag)

    # also crop freq array
    log(f"Cropping Freq array")
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

    
    ## average ##
    log(f"Applying Time Averaging = {par['tN']}", lpf = False)
    log(f"Applying Freq Averaging = {par['fN']}", lpf = False)
    stk_ds = _average(stk = stk, stk_ds = stk_ds, stk_list = stk_list,
                    par = par, err = err_flag, zap = zap_flag)



    # also average freq array
    freq = average(freq, N = par['fN'], nan = zap_flag)


    ## ========================= ##
    ## SAVING DATA TO CONTAINERS ##
    ## ========================= ## 
    for S in stk_list:




        #------ dynamic spectra -------#
        if S in ds_list:
            _ds[S] = stk_ds[S].copy()






        #------- time series -------#
        if S in t_list:
            _t[S] = _scrunch(stk_ds[S], xtype = "t", W = _average_weight(par['fW'], par['fN']), nan = zap_flag)
            if err_flag:

                # for time series error, can average both t and f
                err_ds = average(stk_ds[f"{S}err"].copy(), axis = 1, N = par['tN'])
                err_ds = average(err_ds, axis = 0, N = par['fN'], nan = zap_flag)
                
                # err_ds = f_weight(err_ds, _average_weight(par['fW'], par['fW']))
                _t[f"{S}err"] = _time_err(err_ds, nan = zap_flag)









        #freq spectra
        if S in f_list:
            _f[S] = _scrunch(stk_ds[S], xtype = "f", W = _average_weight(par['tW'], par['tN']))
            if err_flag:
                # for freq error, best to not avearge in time
                err_ds = average(stk_ds[f"{S}err"].copy(), axis = 0, N = par['fN'], nan = zap_flag)
                # err_ds = t_weight(err_ds, par['tW'])
                _f[f"{S}err"] = _freq_err(err_ds, par['t_crop'], par['terr_crop'])
        
        # del temp data
        del stk_ds[S]
        if err_flag:
            del stk_ds[f"{S}err"]
    

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
    if par['f0'] is None:
        log(f"[f0] unspecified, setting f0 = cfreq = {par['cfreq']}", lpf = False)
        par['f0'] = par['cfreq']
    
    return par




def _get_containers_and_lists(data_list):
    """
    Get empty containers and lits
    """

    # Create empty data containers
    _ds, _t, _f = {}, {}, {}

    for S in "IQUV":
        _ds[S], _ds[f"{S}err"] = None, None
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
        print(f"PROCESSING {S}")
        t1 = time()
        # crop on-burst
        stk_ds[S] = pslice(stk[S], *par['t_crop'], axis = 1)        # Time 
        stk_ds[S] = pslice(stk_ds[S], *par['f_crop'], axis = 0)     # Freq
        print(f"Time: {(time() - t1)} s")
        
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
        # average in time, don't want to average in time so as to
        # enable better std calculation
        if par['tN'] > 1:
            stk_ds[S] = average(stk_ds[S], axis = 1, N = par['tN'])
            # if err:
            #     stk_ds[f"{S}err"] = average(stk_ds[f"{S}err"], axis = 1, N = par['tN'])

        if par['fN'] > 1:
            stk_ds[S] = average(stk_ds[S], axis = 0, N = par['fN'], nan = zap)
            # if err:
            #     stk_ds[f"{S}err"] = average(stk_ds[f"{S}err"], axis = 0, N = par['fN'])

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