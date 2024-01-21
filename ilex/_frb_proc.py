## imports
from .data import *
from .logging import log
from .utils import get_stk_from_datalist
from .globals import _G

##==========================##
## MAIN PROCESSING FUNCTION ##
##==========================##

def _proc_data(stk, freq, data_list, par: dict = {}):
    """
    Info:
        Master function for processing data


    Args:
        stk (dict[memmap]): Dictionary of memory maps of Stokes Dynamic spectra
        freq (ndarray): Array of frequencies before cropping/averaging (processing)
        data_list (list): List of requested data products
        par (dict): Parameters for data processing, the following are required
                    [t_crop]: Crop in Time phase (default = [0.0, 1.0])
                    [f_crop]: Crop in Freq phase (default = [0.0, 1.0])
                    [terr_crop]: Crop of Error Time phase (default = None)
                    [tN]: Factor in Time average (default = 1)
                    [fN]: Factor in Freq average (default = 1)
                    [tW]: Time weights (default = None)
                    [fW]: Freq weifhts (default = None)
                    [norm]: Method of normalisation (defualt = "None")
                    [RM]: Rotation Measure (default = 0.0)
                    [cfreq]: Central Frequency (default = 1271.5)
                    [f0]: reference frequency (default = cfreq)
                    [pa0]: PA at f0 (default = 0.0)

    Returns:
        _ds (dict): Dictionary of processed dynamic spectra
        _t (dict): Dictionary of processed time series data
        _f (dict): Dictionary of processed spectra data 
        freq (ndarray): Array of Frequencies in MHz
    
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
    freq = pslice(freq, *par['f_crop'])



    ## weight ##    
    log(f"Applying Freq Weights = {fw_flag}", lpf = False)
    log(f"Applying Time Weights = {tw_flag}", lpf = False)
    stk_ds = _weight(stk = stk, stk_ds = stk_ds, stk_list = stk_list, 
                    par = par, err = err_flag)



    ## normalise ##
    log(f"Applying normalisation = {par['norm']}", lpf = False)
    stk_ds = _normalise(stk = stk, stk_ds = stk_ds, stk_list = stk_list,
                        par = par, err = err_flag)



    ## Channel Zap ## TODO - To be Implemented

    

    ## faraday rotate ##
    if fday_flag:
        log("Applying Faraday De-rotation", lpf = False)
        stk_ds = _faraday_rotate(stk = stk, stk_ds = stk_ds, stk_list = stk_list,
                                freq = freq, par = par, err = err_flag)

    

    ## average ##
    log(f"Applying Time Averaging = {par['tN']}", lpf = False)
    log(f"Applying Freq Averaging = {par['fN']}", lpf = False)
    stk_ds = _average(stk = stk, stk_ds = stk_ds, stk_list = stk_list,
                    par = par, err = err_flag)

    # also average freq array
    freq = average(freq, N = par['fN'])

        

    ## ========================= ##
    ## SAVING DATA TO CONTAINERS ##
    ## ========================= ## 
    for S in stk_list:
        # dynamic spectra
        if S in ds_list:
            _ds[S] = stk_ds[S].copy()
        
        # time series
        if S in t_list:
            _t[S] = np.mean(stk_ds[S], axis = 0)
            if err_flag:
                _t[f"{S}err"] = _time_err(stk_ds[f"{S}err"])

        #freq spectra
        if S in f_list:
            _f[S] = np.mean(stk_ds[S], axis = 1)
            if err_flag:
                _f[f"{S}err"] = _freq_err(stk_ds[f"{S}err"])
        
        # del temp data
        del stk_ds[S]
        if err_flag:
            del stk_ds[f"{S}err"]

        
    # return data containers
    return _ds, _t, _f, freq













# Helper functions for processing
def _proc_par(par):
    """
    Further process parameters
    """

    # check if any are missing
    keys = par.keys()
    parkeys = _G.p.keys()
    metaparkeys = _G.mp.keys()

    for pk in parkeys:
        if pk not in keys:
            par[pk] = _G.p[pk]
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
        # crop on-burst
        stk_ds[S] = pslice(stk[S], *par['t_crop'], axis = 1)        # Time 
        stk_ds[S] = pslice(stk_ds[S], *par['f_crop'], axis = 0)     # Freq
        
        # off-burst
        if err:
            stk_ds[f"{S}err"] = pslice(stk[S], *par['terr_crop'], axis = 1)
            stk_ds[f"{S}err"] = pslice(stk_ds[f"{S}err"], *par['f_crop'], axis = 0)
    
    return stk_ds






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










def _normalise(stk, stk_ds, stk_list, par, err = False):
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
            nconst = np.max(stk[S])
            stk_ds[S] /= nconst
            stk_ds[f"{S}err"] /= nconst

        # normalise by abs max of stokes S
        elif norm == "absmax":
            nconst= np.max(np.abs(stk[S]))
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






def _average(stk, stk_ds, stk_list, par, err = False):
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
            stk_ds[S] = average(stk_ds[S], axis = 0, N = par['fN'])
            if err:
                stk_ds[f"{S}err"] = average(stk_ds[f"{S}err"], axis = 0, N = par['fN'])

    return stk_ds




def _time_err(ds):
    """
    Info:
        Calculate error in Time
    
    Args:
        ds (ndarray): Dynamic spectra

    Returns:
        rms (float): average rms over time
    """
    # scrunch in frequency
    t = np.mean(ds, axis = 0)

    # calculate rms
    rms = np.std(t)

    return rms



def _freq_err(ds):
    """
    Info:
        Calculate error in Freq
    
    Args:
        ds (ndarray): Dynamic spectra

    Returns:
        rms (ndarray): per channel rms 
    """

    # calculate rms
    rms = np.std(ds, axis = 1) / ds.shape[1]**0.5

    return rms




def _zap_chan():
    """
    Info:
        Zap channels
        TODO - To be Implemented
    """
    pass