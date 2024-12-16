##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 25/09/2023 
##
##
## 
## 
## Library of basic functions for analysing FRBs.
##
##===============================================##
##===============================================##
# imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import time
import math
import os, sys
from copy import deepcopy
import inspect
from .pyfit import fit, _posterior
import yaml
from .frbutils import set_dynspec_plot_properties, save_frb_to_param_file

## import utils ##
from .utils import (load_data, save_data, dict_get,
                    dict_init, dict_isall,
                    merge_dicts, dict_null, get_stk_from_datalist, load_param_file, 
                    set_plotstyle, fix_ds_freq_lims)

from .data import *

## import FRB stats ##
from .fitting import (fit_RMquad, fit_RMsynth, RM_QUfit, lorentz,
                     make_scatt_pulse_profile_func)

## import FRB params ##
from .par import FRB_params, FRB_metaparams

# ## import FRB htr functions ##
# from .htr import make_stokes

## import globals ##
from .globals import _G, c

## import plot functions ##
from .plot import (plot_RM, plot_PA, plot_stokes,      
                  plot_poincare_track, create_poincare_sphere, plot_data, _PLOT, plot_dynspec)

## import processing functions ##
from .logging import log, get_verbose, set_verbose, log_title
from .master_proc import master_proc_data
from .widths import *


    

##===============================================##
##                  FRB class                    ##
##===============================================##

class FRB:
    """
    FRB class for Processing of ASKAP FRB data

    Parameters
    ----------
    name: str 
        Name of FRB
    RA: str 
        Right acension position
    DEC: str 
        Declination position
    MJD: float
        Modified Julian date in days
    DM: float 
        Dispersion Measure
    bw: float 
        Bandwidth
    cfreq: float 
        Central Frequency
    t_crop: list 
        Crop start and end phase in Time
    f_crop: list 
        Crop start and end phase in Frequency
    tN: int 
        Factor for averaging in Time
    fN: int 
        Factor for averaging in Frequency
    t_lim: list 
        Limits for FRB in Time
    f_lim: list 
        Limits for FRB in Frequency
    t_lim_base: list
        Base limits of FRB in time (not including t_ref)
    f_lim_base: list
        Base limits of FRB in freq 
    t_ref: float
        Reference zero-point in time
    RM: float 
        Rotation Measure
    f0: float 
        Reference Frequency
    pa0: float 
        Position angle at reference frequency f0
    zapchan: str
        string used for zapping channels, in format -> "850, 860, 870:900" \n
        each element seperated by a ',' is a seperate channel. If ':' is used, user can specify a range of values \n
        i.e. 870:900 -> from channel 870 to 900 inclusive of both.
    verbose: bool 
        Enable verbose logging
    norm: str
        Type of normalisation \n
        [max] - normalise using maximum \n
        [absmax] - normalise using absolute maximum \n
        [None] - Skip normalisation
    terr_crop: list
        bounds for off-pulse region in time [min, max] [ms], default is None
    yaml_file: str
        parameter yaml file of FRB to load in, default is None

    Attributes
    ----------
    par: FRB_params 
        parameters for FRB
    this_par: FRB_params 
        Current instance of 'par'
    prev_par: FRB_params 
        Last instance of 'par'
    metapar: FRB_metaparams 
        hold meta-parameters for FRB
    this_metapar: FRB_metaparams 
        Current instance of 'metapar'
    prev_metapar: FRB_metaparams 
        Last instance of 'metapar'
    ds: Dict 
        Dictionary of loaded stokes dynamic spectra
    pol: Dict 
        Dictionary of loaded Polarisation time series
    _t: Dict 
        Dictionary of cropped stokes time series
    _f: Dict 
        Dictionary of cropped stokes spectra
    _ds: Dict 
        Dictionary of cropped stokes dynamic spectra
    _freq: np.ndarray 
        Cropped Frequency array [MHz]
    _time: np.ndarray
        Cropped time array [ms]
    verbose: bool 
        Enable logging
    savefig: bool 
        Save all created figures to files
    pcol: str 
        Color of text for logging
    empty: bool 
        Variable used to initialise FRB instance and data loading
    plot_type: str
        type of plot \n
        [scatter] - scatter plot with error bars \n
        [lines] - line plot with error patches
    show_plots: bool
        If true, shows plots
    save_plots: bool
        If true, saves plots to file
    residuals: bool
        if true, a residual panel will appear when plotting a fit using pyfit, default is True
    plotPosterior: bool
        if true, will save posterior corner plot when fitting using bayesian method, default is True
    apply_tw: bool
        if true, apply time dependant weights when scrunching in time, i.e making spectra, default is True
    apply_fw: bool
        if true, apply freq dependant weights when scrunching in freq, i.e. making time profiles, default is True
    fitted_params: dict
        dictionary of fitted values, i.e. RM
    dynspec_cmap: str
        cmap for plotting dynamic spectra
    """




    ## [ INITIALISE FRB ] ##
    def __init__(self, yaml_file = None, name: str = _G.p['name'],    RA: str = _G.p['RA'],    DEC: str = _G.p['DEC'], 
                       MJD: float = _G.p['MJD'], DM: float = _G.p['DM'],      bw: int = _G.p['bw'],    cfreq: float = _G.p['cfreq'], 
                       t_crop = None,               f_crop = None,           tN: int = 1,
                       fN: int = 1,                 t_lim_base = _G.p['t_lim_base'],   f_lim_base = _G.p['f_lim_base'],
                       RM: float = _G.p['RM'],      f0: float = _G.p['f0'],  pa0: float = _G.p['pa0'],
                       verbose: bool = _G.hp['verbose'], norm = _G.mp['norm'], dt: float = _G.p['dt'], 
                       df: float = _G.p['df'],      zapchan: str = _G.mp['zapchan'], terr_crop = None, t_ref = _G.p['t_ref']      
                       ):
        """
        Create FRB instance
        """

        self._yaml_file = yaml_file
        
        self.par = FRB_params(name = name, RA = RA, DEC = DEC, MJD = MJD, 
                              DM = DM, bw = bw, cfreq = cfreq,
                              t_lim_base = t_lim_base, f_lim_base = f_lim_base, 
                              RM = RM, f0 = f0, pa0 = pa0, dt = dt, df = df, t_ref = t_ref)

        self.this_par = self.par.copy()
        self.prev_par = FRB_params(EMPTY = True)

        self.metapar = FRB_metaparams(t_crop = t_crop, f_crop = f_crop,
                        terr_crop = terr_crop, tN = tN, fN = fN, norm = norm)


        if t_crop is None:
            self.metapar.t_crop = ["min", "max"]    # crop of time axis
        if f_crop is None:
            self.metapar.f_crop = ["min", "max"]    # crop of frequency axis
            
        self.this_metapar = self.metapar.copy()
        self.prev_metapar = FRB_metaparams(EMPTY = True)


        ## Create data containers
        self.ds = {}                    # container for Dynamic spectra
        self.pol = {}                   # container for polarisation time series data (X, Y)

        for S in "IQUV":
            self.ds[S] = None

        
        ## data instance containers
        self._t = {}                    # container to store time series data
        self._f = {}                    # container to store frequency spectra data
        self._ds = {}                   # container to store dynamic spectra data
        self._freq = {}                 # container to store baseband frequency data
        self._time = {}                 # container to store time samples 

        # initilise data containers
        for S in "IQUVLP":
            self._t[S] = None
            self._t[f"{S}err"] = None
            self._f[S] = None
            self._f[f"{S}err"] = None
        for S in "IQUV":
            self._ds[S]= None
        
  
        self.empty = True               # used to initialise FRB instance and data loading 
        self.verbose = verbose          # TODO: implement
        # set verbose
        set_verbose(self.verbose)

        self.pcol = 'cyan'              # color for verbose printing
        self.plot_type = "scatter"    # type of errorbar plot
        self.residuals = False          # plot residuals when plotting fits
        self.plotPosterior = True      # plot posterior corner plot when plotting fits
        self.save_plots = False
        self.show_plots = True
        self.crop_units = "physical"
        self.zap = False                # if True, will treat arrays as zapped

        # weightings
        self.apply_tW = True                  # apply time weights
        self.apply_fW = True                  # apply freq weights

        self._isinstance = False        # if data instance is valid
        self.fitted_params = {}

        # plotting stuff
        self.dynspec_cmap = "viridis"


        # quick load yaml file
        if yaml_file is not None:
            self.load_data(yaml_file = yaml_file)


    @property
    def dynspec_cmap(self):
        return self._dynspec_cmap

    # Setters
    @dynspec_cmap.setter
    def dynspec_cmap(self, cmap):
        """
        Change cmap of dynamic spectra

        Parameters
        ----------
        cmap : str
            color map

        """
        self._dynspec_cmap = cmap

        set_dynspec_plot_properties(cmap = cmap)



    ##===============================================##
    ##            retrive data funtions              ##
    ##===============================================##

    
    ## [ LOAD IN DATA ] ##
    def load_data(self, dsI: str = None, dsQ: str = None, dsU: str = None, dsV: str = None,
                    yaml_file: str = None, mmap = True, _init = False):
        """
        Load Stokes HTR data

        Parameters
        ----------
        dsI: str 
            Filename of stokes I dynamic spectra
        dsQ: str 
            Filename of stokes Q dynamic spectra
        dsU: str 
            Filename of stokes U dynamic spectra
        dsV: str 
            Filename of stokes V dynamic spectra
        yaml_file: str 
            parameter yaml file for FRB, default is None
        mmap: bool 
            Enable memory mapping for loading
        _init: bool 
            For initial Data loading
        """

        self._yaml_file = yaml_file

        log_title("Loading in Stokes dynamic spectra. Assuming the data being loaded are .npy files", col = "lblue")


        if yaml_file is not None:
            log("Loading from yaml file", lpf_col = self.pcol)

            # load pars
            yaml_pars = load_param_file(yaml_file)

            # extract pars
            pars = merge_dicts(yaml_pars['par'], yaml_pars['metapar'], yaml_pars['hyperpar'])
            self.set(**pars)

            # set weights if given
            self.par.set_weights(xtype = "t", **yaml_pars['weights']['time'])
            self.par.set_weights(xtype = "f", **yaml_pars['weights']['freq'])

            # set loaded files
            dsI, dsQ = yaml_pars['data']['dsI'], yaml_pars['data']['dsQ']
            dsU, dsV = yaml_pars['data']['dsU'], yaml_pars['data']['dsV']

            # check if plotstyle file is given
            set_plotstyle(yaml_pars['plots']['plotstyle_file'])
            if yaml_pars['plots']['plotstyle_file'] is None:
                log("Setting plotting style: Default")
            else:
                log(f"setting plotting style: {yaml_pars['plots']['plotstyle_file']}")


        def init_par_from_load(x):
            """
            Initialise a number of parameters from loaded file
            """

            self.par.nchan = x.shape[0]                     # assumed that dyn spec is [freq,time]
            self.par.nsamp = x.shape[1]       
            self.par.t_lim_base  = [0.0, self.par.dt * self.par.nsamp]



        ## dict. of files that will be loaded in
        data_files = {"dsI": dsI, "dsQ": dsQ, "dsU": dsU, "dsV": dsV}
        old_chans = None

        # loop through files
        for key in data_files.keys():

            file = data_files[key]
            init_key = None
            if file is not None:

                # load all dynamic spectra
                self.ds[key[-1]] = load_data(file, mmap)
                log(f"Loading stokes {key[-1]} Dynspec from: {file} with shape {self.ds[key[-1]].shape}",
                        lpf_col=self.pcol)

                if init_key is None:
                    init_par_from_load(self.ds[key[-1]])
                    init_key = key
            
                # check if any channels are nan's i.e. flagged
                chans = self.ds[key[-1]][:,0]
                if np.any(np.isnan(chans)):
                    self.zap = True
                    log("Finding zapped channels...")
                    self.metapar.zapchan = get_zapstr(chans, self.par.get_freqs())
                    if old_chans is not None:
                        if not np.all(old_chans == chans):
                            log("Channels being zapped are different for each Stokes Dynamic spectra!!", stype = "warn")
                    old_chans = chans.copy()








        

        
    ## [ SAVING FUNCTION - SAVE CROP OF DATA ] ##
    def save_data(self, data_list = None, name = None, save_yaml = False, yaml_file = None, **kwargs):
        """
        Save current instance data

        Parameters
        ----------
        data_list : List(str), optional
            List of data to save, by default None
        name : str, optional
            Common Pre-fix for saved data, by default None, if None the name parameter of the
            FRB class will be used.
        """

        log_title("Saving Stokes data, the data is saved as .npy files. ", col = "lblue")

        if save_yaml:
            log("Saving fitted parameters to yaml file...", lpf_col = "green")
            save_frb_to_param_file(self, yaml_file)


        if data_list is None:
            log("No data specified for saving...", stype = "warn")
            return


        print("Saving the following data products:")
        for data in data_list:
            print(f"[{data}]")

        # get data
        pdat = self.get_data(data_list, get = True)
        if not self._isdata():
            return 
        

        if name is None:
            frbname = self.par.name
            if frbname is None:
                frbname = "FRBXXXXXX"
            name = os.path.join(os.getcwd(), frbname)

        # save data
        for data in pdat.keys():
            np.save(name + f"_{data}.npy", pdat[data])

        return




    def set(self, **kwargs):
        """
        Set FRB parameters, see class parameters
        """

        # update pars
        par = self._from_kwargs_get_par(**kwargs)
        self.par.set_par(**par)

        # update metapars
        metapar = self._from_kwargs_get_metapar(**kwargs)
        self.metapar.set_metapar(**metapar)

        # update hyperpars
        self._update_hyperpar(**kwargs)        






    def get_freqs(self):
        """
        Get Frequencies
        
        """

        if self.empty:
            return self.par.get_freqs()
        else:
            return self.this_par.get_freqs()




        
    # implement FRB_params struct  
    ## [ GET DATA ] ##
    def get_data(self, data_list = "dsI", get = False, ignore_nans = False, debias = False, 
                    ratio = False, ratio_rms_threshold = None, **kwargs):
        """
        Make new instance of loaded data. This will take a crop of the 
        loaded mmap-ed stokes data, pass it through the back-end processing
        function and save the data in memory in the ._ds, _t, _f, _time and _freq
        class instance attributes.

        Parameters
        ----------
        data_list : List(str) or str, optional
            List of data products to load in, by default "dsI"
        get : bool, optional
            Return new crops of data, by default False and will only save data
            instances to class container attributes
        ignore_nans : bool, optional
            If true, if nans exist in data, they will be removed before saving the data instance
        debias, bool, optional
            If true, tL/fL and tP/fP will be debiased 
        ratio, bool, optional
            If true, calculate X/I for t and f products
        ratio_rms_threshold : float, optional
            Mask X/I data by ratio_rms_threshold * rms

        Returns
        -------
        data: Dict, optional
            Dictionary of processed data crops, by default None if get = False
        """

        log_title("Retrieving Processed Data products. Any currently loaded crops of data will be overwritten. ", col = "lblue")

        # update par and metapar if nessesary
        self._load_new_params(**kwargs)

        
        # process data_list as str
        if type(data_list) == str:
            if data_list == "all":
                data_list = _G.hkeys
            else:
                data_list = [data_list]
                
        log(f"Retrieving the following data: {data_list}", lpf_col = self.pcol)

        # get all data products needed
        data_products = self._init_proc(data_list, debias = debias, ratio = ratio)

        ## first check if there is data to use
        if not self._isvalid(data_products):
            log("Loaded data not avaliable or incorrect DS shapes", stype = "err",
                lpf_col = self.pcol)
            self._isinstance = False
            return 

        ## make new instances
        self._make_instance(data_list = data_list, ignore_nans = ignore_nans, debias = debias, ratio = ratio,
                            ratio_rms_threshold = ratio_rms_threshold)


        ## set new instance param 
        self._save_new_params()
        

        self._isinstance = True

        # check if get is true
        if get:
            # return instance
            return self._get_instance(data_list, ignore_nans)


        #return data
        return




    def _get_instance(self, data_list = None, ignore_nans = False):
        """
        retrieve data products

        Parameters
        ----------
        data_list : List(str), optional
            crop types to return, by default None

        Returns
        -------
        data: Dict
            Dictionary of data crops
        """
        # initialise new data list
        new_data = {}

        # ingore nans? -> make mask for this process
        f_mask = np.ones(self._freq.size, dtype = bool)
        if ignore_nans:
            # find first data that isn't None
            while True:
                for key in self._ds.keys():
                    if self._ds[key] is not None:
                        f_mask[np.isnan(self._ds[key][:,0])] = False
                        break

                for key in self._f.keys():
                    if self._f[key] is not None:
                        f_mask[np.isnan(self._f[key])] = False
                        break

                # if no freq data, just pass through 
                break



        # flags
        err_flag = self._iserr()

        for data in data_list:
            stk = data[-1]
            # dynamic spectra
            if "ds" in data:
                new_data[data] = self._ds[stk][f_mask,:].copy()

            # time series
            elif "t" in data:
                new_data[data] = self._t[stk].copy()
                new_data[f"{data}err"] = self._t[f"{stk}err"]

            # frequency spectra
            elif "f" in data:
                new_data[data] = self._f[stk][f_mask].copy()
                new_data[f"{data}err"] = self._f[f"{stk}err"]

        # also add freqs
        if self._freq is not None:
            new_data['freq'] = self._freq[f_mask].copy()
        else:
            log("Couldn't get freq array, something went wrong", stype = "warn")

        # also add times
        if self._time is not None:
            new_data['time'] = self._time.copy()
        else:
            log("Couldn't get time array, something went wrong", stype = "warn")
        

        return new_data



    def _make_instance(self, data_list = None, ignore_nans = False, debias = False, ratio = False,
                        ratio_rms_threshold = None):
        """
        Make New data crops for current instance

        Parameters
        ----------
        data_list : List(str), optional
            List of crop products to make, by default None
        """

        # assuming all prior checks on data were successful

        # purge everything
        for S in"IQUV":
            self._ds[S] = None
            self._ds[f"{S}err"] = None
            self._t[S] = None
            self._t[f"{S}err"] = None
            self._f[S] = None
            self._f[f"{S}err"] = None
            self._freq = None
            self._time = None
        for S in "LP":
            self._t[S] = None
            self._t[f"{S}err"] = None
            self._f[S] = None
            self._f[f"{S}err"] = None


        # get frequencies
        freqs = self.par.get_freqs()
        

        # set up parameter dictionary
        full_par = merge_dicts(self.this_metapar.metapar2dict(), 
                                 self.this_par.par2dict())

        # get tW and fW
        temp_w_par = self.par.copy()
        temp_w_par.update_from_crop(t_crop = full_par['t_crop'],
                                        f_crop = full_par['f_crop'])

        if self.apply_tW:
            log("Retrieving Time Weights")
            log("=======================")
            full_par['tW'] = self.par.tW.get_weights(x = temp_w_par.get_times())
            log(temp_w_par.tW, lpf = False)
        if self.apply_fW:
            log("Retrieving Freq Weights")
            log("=======================")
            full_par['fW'] = self.par.fW.get_weights(x = temp_w_par.get_freqs())
            log(temp_w_par.fW, lpf = False)


        # pass through to backend processing script
        _ds, _t, _f, self._freq, _flags = master_proc_data(self.ds, freqs, 
                                            data_list, full_par, debias, ratio, ratio_rms_threshold)

        # process flags
        self.zap = _flags['zap_flag']

        # ingore nans? -> make mask for this process
        f_mask = np.ones(self._freq.size, dtype = bool)
        if ignore_nans and self.zap:
            # find first data that isn't None
            while True:
                for key in _ds.keys():
                    if _ds[key] is not None:
                        f_mask[np.isnan(_ds[key][:,0])] = False
                        break

                for key in _f.keys():
                    if _f[key] is not None:
                        f_mask[np.isnan(_f[key])] = False
                        break

                # if no freq data, just pass through 
                break
         

        log("Saving new data products to latest instance", lpf_col = self.pcol)

        aval_key_for_time = None
        _timesize = 0
        # dynspecs
        ds_list = _ds.keys()
        for key in ds_list:
            if _ds[key] is not None:
                if "err" not in key:
                    aval_key_for_time = key
                    _timesize = _ds[key].size
                self._ds[key] = _ds[key][f_mask,:].copy()
                _ds[key] = None
        
        # time series
        t_list = _t.keys()
        for key in t_list:
            if _t[key] is not None: 
                if "err" not in key:
                    aval_key_for_time = key
                    _timesize = _t[key].size
                self._t[key] = _t[key].copy()
                _t[key] = None
        
        # freq spectra
        f_list = _f.keys()
        for key in f_list:
            if _f[key] is not None:
                self._f[key] = _f[key][f_mask].copy()
                _f[key] = None
        
        # proc freq array, nan 
        self._freq = self._freq[f_mask]

        if aval_key_for_time is not None:
            self._time = self.this_par.get_times()

        return


    
    def _clear_instance(self, data_list = None):
        """
        Remove specified data products of crops

        Parameters
        ----------
        data_list : List(str), optional
            List of data products to clear, by default None
        """

        # flags
        err_flag = self._iserr()

        if data_list is None:
            data_list = _G.hkeys[:-2]
            # remove freqs
            self._freq = None

        log(f"Clearing data: {data_list}")

        for data in data_list:
            stk = data[-1]
            # dynamic spectra
            if "ds" in data:
                self._ds[stk] = None
            
            # time series
            elif "t" in data:
                self._t[stk] = None
                if err_flag:
                    self._t[f"{stk}err"] = None

            # spectra
            elif "f" in data:
                self._f[stk] = None
                if err_flag:
                    self._f[f"{stk}err"] = None

        if "freq" in data_list:
            # remove freqs
            self._freq = None

        if "time" in data_list:
            # remove time samples
            self._time = None

        return



    
    def _init_proc(self, data_list, debias = False, ratio = False):
        """
        Check if all requested data products and their
        dependencies are being requested. 

        Parameters
        ----------
        data_list: List(str)
            List of requested cropped data products
        """

        # get stokes data to load in
        stk = get_stk_from_datalist(data_list)

        # check if Q or U is there, and if RM is non-zero, if so
        # load both Q and U
        if (("Q" in stk) != ("U" in stk)) and (self.this_par.RM is not None):
            # add missing stokes to stk list
            if "Q" in stk:
                log("Added Stokes U to process for RM correction", lpf = False)
                stk += "U"
            else:
                log("Added Stokes Q to process for RM correction", lpf = False)
                stk += "Q"
        

        # if norm == "I", then we want to normalise all data using "I", this
        # must add it to the list.
        if self.this_metapar.norm == "maxI":
            log("Added stokes I for normalisation purposes", lpf = False)
            stk += "I"


        # if requesting L and/or P
        if ("tL" in data_list) or ("fL" in data_list):
            for s in "QU":
                if s not in stk:
                    log(f"Added stokes {s} to process for retrieving L polarisation", lpf = False)
                    stk += s
        
        if ("tP" in data_list) or ("fP" in data_list):
            for s in "QUV":
                if s not in stk:
                    log(f"Added stokes {s} to process for retrieving P polarisation", lpf = False)
                    stk += s

        # if debiasing L and/or P
        add_stokes_I = False
        for s in ["tL", "fL", "tP", "fP"]:
            if s in data_list:
                if debias:
                    add_stokes_I = True
        if add_stokes_I:
            log("Added stokes I to process for debiasing L and/or P polarisations", lpf = False)
            if "I" not in stk:
                stk += "I"
        
        # if calculating ratios
        if ratio:
            log("Added stokes I to process for calculating stokes ratios", lpf = False)
            if "I" not in stk:
                stk += "I"
        
        return stk



            





    

    ##===============================================##
    ##            validate par functions             ##
    ##===============================================##


    def _update_par(self, **kwargs):
        """
        Info:
            Return a copy of FRB_params class with updated parameters

        Args:
            **kwargs

        """
        # extract pars
        par = self._from_kwargs_get_par(**kwargs)
        
        # create copy of par
        self.this_par = self.par.copy()

        # update copy
        self.this_par.set_par(**par)

        # update from crop
        metapar = self._from_kwargs_get_metapar(**kwargs)
        self.this_par.update_from_crop(metapar['t_crop'], metapar['f_crop'],
                                      metapar['tN'], metapar['fN'])
        
        
        
        


    def _update_metapar(self, **kwargs):
        """
        Info:
            Return a copy of FRB_metaparams class with updated parameters

        Args:
            **kwargs
        """
        # extract metapars
        metapar = self._from_kwargs_get_metapar(**kwargs)

        # create copy of par
        self.this_metapar = self.metapar.copy()

        # update
        self.this_metapar.set_metapar(**metapar)



    def _update_hyperpar(self, **kwargs):
        """
        Info:
            Return updated hyper params

        Args:
            **kwargs
        """
        hyperpar = {}
        for key in _G.hp.keys():
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
            
        # set verbose
        set_verbose(self.verbose)
            



    def _load_new_params(self, **kwargs):
        """
        Update parameters with keywords for current instance

        """  
        # make copy of kwargs that change, that way changes that are made do 
        # not propagate 
        kwargs_keys = kwargs.keys()

        # check if any are already made static
        static_keys = []
        for key, item in _G.sp.items():
            if key in kwargs_keys:
                if item not in kwargs_keys:
                    static_keys += [key]
                else:
                    kwargs[key] = deepcopy(kwargs[item])

        kwargs = {_G.sp[k] if k in static_keys else k: v for k, v in kwargs.items()}

        # add copies of items back in, these will be processed through without touching the
        # original ones
        for static_key in static_keys:
            kwargs[static_key] = deepcopy(kwargs[_G.sp[static_key]])


        # copy over current hyperparams to kwargs
        metapar = self.metapar.metapar2dict()
        kw = kwargs.keys()
        for key in metapar.keys():
            if key not in kw:
                kwargs[key] = metapar[key]       
                
        # make sure metaparameters are updated first  
        self._proc_kwargs(**kwargs) 

        # update hyper parameters
        self._update_hyperpar(**kwargs)

        # update self.this_metapar
        self._update_metapar(**kwargs)

        # update self.this_par
        self._update_par(**kwargs)







    

    def _save_new_params(self):
        """
        save Current instance of FRB_params and FRB_metaparams

        """

        # update self.prev_par
        self.prev_par = self.this_par.copy()

        # update self.prev_metapar
        self.prev_metapar = self.this_metapar.copy()



    
    def _from_kwargs_get_par(self, **kwargs):
        """
        Info:
            Get all parameters from kwargs dictionary,
            missing parameters will be taken from [self.par]

        Args:
            **kwargs
        
        """
        par = {}
        base_par = self.par.par2dict()

        for key in _G.p.keys():
            # check if key part of par list
            if key in kwargs.keys():
                par[key] = kwargs[key]
            else:
                par[key] = base_par[key]


        return par



    def _from_kwargs_get_metapar(self, **kwargs):
        """
        Info:
            Get all Meta parameters from kwargs dictionary,
            missing meta parameters will be taken from [self.metapar]

        Args:
            **kwargs

        """
        meta_par = {}
        base_metapar = self.metapar.metapar2dict()

        for key in _G.mp.keys():
            # check if key part of meta par list
            if key in kwargs.keys():
                meta_par[key] = kwargs[key]
            else:
                meta_par[key] = base_metapar[key]

        return meta_par



    def _proc_kwargs(self, **kwargs):
        """
        Process Kwargs
        """
        keys = kwargs.keys()

        if self.crop_units not in ["physical", "phase"]:
            log("Units for cropping must be one of: ['physical', 'phase'] ", stype = "err")
            return

        def check_crop_for_str(crop, domain):
            """ Check for crop "min" and "max" specifiers"""
            if domain == "t":
                _vars = self.par.t_lim
            elif domain == "f":
                _vars = self.par.f_lim
            else:
                log("Something went wrong converting crops, no domain chosen.", stype = "err")

            phase_vars = [0.0, 1.0]
            if self.crop_units == "physical":
                phase_vars = [_vars[0], _vars[1]]

            for i, spe in zip([0, -1], ["min", "max"]):
                if isinstance(crop[i], str):
                    if crop[i] == spe:
                        # check if other crop comp is phase or ms
                        if isinstance(crop[i+1], float) or isinstance(crop[i+1], int):
                            if crop[i+1] > 1.0:
                                # convert to min/max
                                crop[i] = _vars[i]
                            else:
                                crop[i] = phase_vars[i]
                        elif isinstance(crop[i+1], str):
                            crop[i] = phase_vars[i]
                        else:
                            log(f"Typing of crop isn't right. {crop[i+1]}", stype = "err")
                    else:
                        log("Incorrect placement of crop specifiers, must be ['min', 'max'] if being used.", stype = "err")
                elif isinstance(crop[i], float) or isinstance(crop[i], int):
                    pass
                else:
                    log(f"Typing of crop isn't right. {crop[i]}", stype = "err")
            return crop[0], crop[1]
        


        # check if t_crop has been given in units of ms
        if "t_crop" in keys:
            
            kwargs['t_crop'][0], kwargs['t_crop'][1] = check_crop_for_str(kwargs['t_crop'], "t")

            if self.crop_units == "physical":

                prev_t = kwargs['t_crop'].copy()
                new_t,_ = self.par.lim2phase(t_lim = prev_t, snap = True)
                kwargs['t_crop'][0], kwargs['t_crop'][1] = new_t[0], new_t[1]

                if kwargs['t_crop'][0] < 0.0: kwargs['t_crop'][0] = 0.0
                if kwargs['t_crop'][1] > 1.0: kwargs['t_crop'][1] = 1.0

                log(f"Converting Time crop {prev_t} ms -> {kwargs['t_crop']} phase units", lpf = False)
            
            elif self.crop_units == "phase":
                # check if within phase units
                bad_crop_flag = False
                prev_t = kwargs['t_crop'].copy()
                if (kwargs['t_crop'][0] < 0.0) or (kwargs['t_crop'][0] > 1.0):
                    bad_crop_flag = True
                    kwargs['t_crop'][0] = 0.0
                if (kwargs['t_crop'][1] < 0.0) or (kwargs['t_crop'][1] > 1.0):
                    bad_crop_flag = True
                    kwargs['t_crop'][1] = 1.0

                if bad_crop_flag:
                    log(f"Phase crop in time was out-of-bounds of [0.0, 1.0], setting: [{prev_t[0]}, {prev_t[1]}] -> [{kwargs['t_crop'][0]},{kwargs['t_crop'][1]}]")


        # check if t_crop has been given in units of ms
        if "f_crop" in keys:

            kwargs['f_crop'][0], kwargs['f_crop'][1] = check_crop_for_str(kwargs['f_crop'], "f")

            if kwargs['f_crop'][0] > 1.0 or kwargs['f_crop'][1] > 1.0:
                prev_f = kwargs['f_crop'].copy()
                _, new_f = self.par.lim2phase(f_lim = prev_f, snap = True)
                kwargs['f_crop'][0], kwargs['f_crop'][1] = new_f[0], new_f[1]

                if kwargs['f_crop'][0] < 0.0: kwargs['f_crop'][0] = 0.0
                if kwargs['f_crop'][1] > 1.0: kwargs['f_crop'][1] = 1.0

                log(f"Converting Freq crop {prev_f} MHz -> {kwargs['f_crop']} phase units", lpf = False)

        # check if terr_crop has been given in units of ms
        if "terr_crop" in keys:
            if kwargs['terr_crop'] is not None:
                
                kwargs["terr_crop"][0], kwargs["terr_crop"][1] = check_crop_for_str(kwargs["terr_crop"], "t")

                if self.crop_units == "physical":
                    prev_t = kwargs['terr_crop'].copy()
                    new_t,_ = self.par.lim2phase(t_lim = prev_t, snap = True)
                    kwargs['terr_crop'][0], kwargs['terr_crop'][1] = new_t[0], new_t[1]

                    if kwargs['terr_crop'][0] < 0.0: kwargs['terr_crop'][0] = 0.0
                    if kwargs['terr_crop'][1] > 1.0: kwargs['terr_crop'][1] = 1.0

                    log(f"Converting err Time crop {prev_t} ms -> {kwargs['terr_crop']} phase units", lpf = False)
                
                elif self.crop_units == "phase":
                    # check if within phase units
                    bad_crop_flag = False
                    prev_t = kwargs['terr_crop'].copy()
                    if (kwargs['terr_crop'][0] < 0.0) or (kwargs['terr_crop'][0] > 1.0):
                        bad_crop_flag = True
                        kwargs['terr_crop'][0] = 0.0
                    if (kwargs['terr_crop'][1] < 0.0) or (kwargs['terr_crop'][1] > 1.0):
                        bad_crop_flag = True
                        kwargs['terr_crop'][1] = 1.0

                    if bad_crop_flag:
                        log(f"Phase error crop in time was out-of-bounds of [0.0, 1.0], setting: [{prev_t[0]}, {prev_t[1]}] -> [{kwargs['terr_crop'][0]},{kwargs['terr_crop'][1]}]")
        
            



    



    ## [ CHECK IF DATA PROUCTS ARE VALID ] ##
    def _isvalid(self, data_products: list = None):
        """
        Check if data products are valid, are they loaded? Do their shapes match

        Parameters
        ----------
        data_products : list(str), optional
            Data products to check against, by default None

        Returns
        -------
        bool
            0 if failed, 1 if passed
        """

        data_shape = []
        print(data_products)
        for key in data_products:
            # check if none
            if self.ds[key] is None:
                log(f"Missing data for [{key}]")
                return 0
            
            data_shape.append(list(self.ds[key].shape))
        
        # check if shape of all data matches
        if not all(x==data_shape[0] for x in data_shape):
            log("Data shape mismatch between loaded Dynamic spectra")
            return 0

        return 1



    def _iserr(self):
        """
        Check if off-pulse region crop parameters, i.e. terr_crop
        has been given, if so the off-pulse rms will be calculated.

        """

        return self.this_metapar.terr_crop is not None
    

    def _isdata(self):

        return self._isinstance






    ##===============================================##
    ##             Diagnostic functions              ##
    ##===============================================##


    def __str__(self):
        """
        Info:
            Print info about FRB class

        """
        
        #create string outlining parameters

        outstr = ""
        outstr += "="*80 + "\n"
        outstr += " "*32 + " Crop Parameters " + ""*32 + "\n"
        outstr += "="*80 + "\n\n"
        outstr += "PARAMETERS:".ljust(25) + "SAVED:".ljust(25) + "INST:".ljust(25) + "\n"
        outstr += "="*80 + "\n\n"

        for _,key in enumerate(_G.mp.keys()):
            val = getattr(self.metapar, key)
            val2 = getattr(self.prev_metapar, key)
            
            outstr += f"{key}:".ljust(25) + f"{val}".ljust(25) + f"{val2}".ljust(25) + "\n"


        outstr += "\n"
        outstr += "="*80 + "\n"
        outstr += " "*32 + " Data Parameters " + " "*32 + "\n"
        outstr += "="*80 + "\n\n"
        outstr += "PARAMETERS:".ljust(25) + "SAVED:".ljust(25) + "INST:".ljust(25) + "\n"
        outstr += "="*80 + "\n\n"

        for _,key in enumerate(_G.p.keys()):
            val = getattr(self.par, key)
            val2 = getattr(self.prev_par, key)

            outstr += f"{key}:".ljust(25) + f"{val}".ljust(25) + f"{val2}".ljust(25) + "\n"

        
        # outline loaded data
        outstr += "\n"
        outstr += "="*80 + "\n"
        outstr += " "*30 + "    DATA products   " + " "*30 + "\n"
        outstr += "="*80 + "\n\n"
        outstr += "TYPE:".ljust(25) + "SHAPE:".ljust(25) + "\n"
        outstr += "="*80 + "\n\n"
        
        for S in "IQUV":
            if self.ds[S] is not None:
                outstr += f"{S}:".ljust(25) + f"{list(self.ds[S].shape)}".ljust(25) + "\n"
        
        #now print data instance
        outstr += "\n"
        outstr += "="*80 + "\n"
        outstr += " "*30 + "    DATA instance   " + " "*30 + "\n"
        outstr += "="*80 + "\n\n"
        outstr += "TYPE:".ljust(25) + "SHAPE/VAL:".ljust(25) + "\n"
        outstr += "="*80 + "\n\n"
        ds_str = ""
        t_str = "\n"
        f_str = "\n"
        for S in "IQUV":
            if self._ds[S] is not None:
                ds_str += f"ds{S}:".ljust(25) + f"{list(self._ds[S].shape)}".ljust(25) + "\n"
        
        for S in "IQUVLP":
            
            if self._t[S] is not None:
                t_str += f"t{S}:".ljust(25) + f"{list(self._t[S].shape)}".ljust(25) + "\n"
            if self._t[f"{S}err"] is not None:
                Serr = f"{S}err"
                t_str += f"t{S}err:".ljust(25) + f"{self._t[Serr]}".ljust(25) + "\n"
            
            if self._f[S] is not None:
                f_str += f"f{S}:".ljust(25) + f"{list(self._f[S].shape)}".ljust(25) + "\n"
            if self._f[f"{S}err"] is not None:
                Serr = f"{S}err"
                f_str += f"f{S}err:".ljust(25) + f"{list(self._f[Serr].shape)}".ljust(25) + "\n"
        
        outstr += ds_str + t_str + f_str

        if self._freq is not None and len(self._freq) > 0:
            outstr += f"freqs:".ljust(25) + f"top:{self._freq[0]}, bottom:{self._freq[-1]}" + "\n\n"

        
        def _print_fitted_params(pstr, pars):

            for i, key in enumerate(pars.keys()):
                pstr += f"{key}:".ljust(25) + f"{pars[key].val}".ljust(25) + f"+{pars[key].p}".ljust(20) + f"-{pars[key].m}".ljust(20) + "\n"

            return pstr


        # print fitted params 
        if len(self.fitted_params) > 0:
            
            outstr += "\n"
            outstr += "="*80 + "\n"
            outstr += " "*31 + " Fitted Parameters " + " "*31 + "\n"
            outstr += "="*80 + "\n\n"
            outstr += "PARAMETERS:".ljust(25) + "VALUES:".ljust(25) + "+ ERR:".ljust(20) + "- ERR:".ljust(20) + "\n"
            outstr += "="*80 + "\n\n"

            keys = self.fitted_params.keys()

            if "RM" in keys:
                outstr += "#"*20 + "\n"
                outstr += "      Fitted RM      \n"
                outstr += "#"*20 + "\n"
                outstr = _print_fitted_params(outstr, self.fitted_params['RM'])

            if "tscatt" in keys:
                outstr += "#"*20 + "\n"
                outstr += "    Fitted tscatt    \n"
                outstr += "#"*20 + "\n"
                outstr = _print_fitted_params(outstr, self.fitted_params['tscatt']) 

            if "scintband" in keys:
                outstr += "#"*20 + "\n"
                outstr += "   Fitted scintband  \n"
                outstr += "#"*20 + "\n"
                outstr = _print_fitted_params(outstr, self.fitted_params['scintband']) 



        return outstr



    
























    ##===============================================##
    ##            Further FRB processing             ##
    ##===============================================##


    ## [ FIND FRB PEAK AND TAKE REGION AROUND IT ] ##
    def find_frb(self, method = "sigma", mode = "median", sigma: int = 5, rms_guard: float = 10, rms_width: float = 50, 
                    rms_offset: float = 60, yfrac: float = 0.95, buffer: float = None, 
                    padding: float = None, dt_from_peak_sigma: float = None, **kwargs):
        """
        This function uses a number of method of finding the bounds of a burst.

        1. Find FRB bounds using a sigma threshold [method = "sigma"]
        find_optimal_sigma_width(sigma, rms_guard, rms_width, rms_offset)

        2. Find FRB width and centroid using a fractional fluence threshold method [method = "fluence"]
        find_optimal_fluence_width(yfrac)

        Note, the centroid of the burst is the point along the burst that splits the fluence 50/50 on either side.


        Parameters
        ----------
        method: str
            method to use for finding burst bounds ["sigma", "fluence"]
        mode: str
            type of algorithm to use when finding optimal fluence width (method = "fluence")\n
            [median] -> find burst width by estimating centroid of burst and fluence threshold on either side \n
            [min] -> find minimum burst width that captures the desired fluence threshold (moving window algorithm) 
        sigma: int 
            S/N threshold
        rms_guard: float 
            gap between estiamted pulse region and 
            off-pulse region for rms and baseband estimation, in (ms)
        rms_width: float 
            width of off-pulse region on either side of pulse region in (ms)
        rms_offset: float 
            rough offset from peak on initial S/N threshold in (ms)
        yfrac: float
            fraction of total fluence on either side of FRB effective centroid to take
            as FRB bounds
        buffer: float
            initial width of data in [ms] centered at the peak that will be used to estimate 
            FRB bounds
        pading: float
            Add additional padding to measured bounds, as a fraction of the width of the burst
        dt_from_peak_sigma: float
            Determine maximum time resolution (dt) to achieve a peak S/N of dt_from_peak_sigma
        **kwargs: 
            FRB parameters + FRB meta-parameters

        Returns
        -------
        t_crop: list
            New Phase start and end limits for found frb burst
        t_ref: float
            Zero point, either the peak or centroid, depending on the method used

        """
        log_title(f"Looking for FRB burst.", col = "lblue")
        ms2phase = lambda x : x / (tI.size * self.this_par.dt)

        ##====================##
        ## check if data valid##
        ##====================## 

        kwargs['t_crop'] = ["min", "max"]
        # kwargs['f_crop'] = ["min", "max"]  
        kwargs['terr_crop'] = None
        
        tN = None
        if dt_from_peak_sigma is not None:
            kwargs['tN'] = 1

            # get full data
            self.get_data("tI", **kwargs)
            if not self._isdata():
                return None
            tI = self._t['I']

            kwargs['tN'] = find_optimal_sigma_dt(tI, sigma = dt_from_peak_sigma,
                        rms_offset = ms2phase(rms_offset), rms_width = ms2phase(rms_width))

            tN = kwargs['tN']
            
        # get data   
        self.get_data("tI", **kwargs)
        if not self._isdata():
            return None

        
        # make smaller buffer of data
        if buffer is not None:
            log(f"Searching Buffer [{buffer}] ms around peak of burst", lpf_col = self.pcol)
            buffer = int(buffer / self.this_par.dt)
            peak = np.argmax(self._t['I'])

            tI = self._t['I'][peak - buffer//2 : peak + buffer//2]
            buffer_ref = peak - buffer//2
        else:
            log(f"Searching full time series", lpf_col = self.pcol)
            tI = self._t['I']
            buffer_ref = 0

        

        # now choose method of finding burst bounds
        if method == "sigma":
            ms2phase = lambda x : x / (tI.size * self.this_par.dt)
            ref_ind, lw, rw = find_optimal_sigma_width(tI = tI, sigma = sigma,
                                rms_guard = ms2phase(rms_guard), 
                                rms_width = ms2phase(rms_width),
                                rms_offset = ms2phase(rms_offset))
            
            log("Setting zero point reference [t_ref] to PEAK of burst", lpf_col = self.pcol)
        
        elif method == "fluence":
            ref_ind, lw, rw = find_optimal_fluence_width(tI = tI, yfrac = yfrac, mode = mode)

            log("Setting zero point reference [t_ref] to EFFECTIVE CENTROID of burst", lpf_col = self.pcol)
        
        else:
            log(f"Undefined method [{method}].. Aborting!", lpf_col = self.pcol, stype = "err")
            return (None)*2


        # Calculate new t_crop and t_ref relative to full time series dataset
        t_ref = (buffer_ref + ref_ind) * self.this_par.dt
        t_crop = [-lw * self.this_par.dt, rw * self.this_par.dt]


        # add padding 
        width = t_crop[1] - t_crop[0]
        padded_width = width
        if padding is not None:
            t_crop[0] -= padding * width
            t_crop[1] += padding * width
            padded_width += 2 * padding * width
        else:
            padding = 0


        # if units are physical 
        if self.crop_units == "phase":
            self.par.set_par(t_ref = t_ref)
            t_crop,_ = self.par.lim2phase(t_lim = t_crop)

        self.metapar.set_metapar(t_crop = t_crop)
        self.metapar.set_metapar(terr_crop = [-rms_offset - rms_width, -rms_offset])
        if dt_from_peak_sigma is not None:
            self.metapar.set_metapar(tN = kwargs['tN'])
        self.par.set_par(t_ref = t_ref)

        print("New t_crop: [{:.4f}, {:.4f}]".format(t_crop[0],t_crop[1]))
        print(f"Setting terr_crop: [{-rms_offset - rms_width:.4f}, {-rms_offset:.4f}]")
        print(f"New time series 0-point: [{t_ref:.4f}]")        
        if dt_from_peak_sigma is not None:
            print(f"time resolution for peak S/N [{dt_from_peak_sigma:.4f}]: {self.this_par.dt:.4f} ms (tN = {kwargs['tN']})")
        log(f"Width of burst without padding: {width:.4f} ms", lpf_col = self.pcol)
        log(f"Width LHS, RHS of zero point (without padding): {-lw * self.this_par.dt:.4f}, {rw * self.this_par.dt:.4f} ms", 
                                                                                                lpf_col = self.pcol)
        log(f"Width of burst with padding: {padded_width:.4f} ms", lpf_col = self.pcol)
        log(f"Width LHS, RHS of zero point (with padding): {-lw * self.this_par.dt - padding * width:.4f}, {rw * self.this_par.dt + padding * width:.4f} ms", 
                                                                                                lpf_col = self.pcol)


        # clear dsI
        self._clear_instance(data_list = ["dsI"])

        return t_crop, t_ref, tN
    










    ##===============================================##
    ##                Plotting Methods               ##
    ##===============================================##
    def plot_data(self, data = "dsI", ax = None, debias = False, ratio = False, ratio_rms_threshold = None,
                     filename: str = None, **kwargs):
        """
        General Plotting function, choose to plot either dynamic spectrum or time series 
        data for all stokes parameters

        Parameters
        ----------
        data : str, optional
            type of data to plot, by default "dsI"
        ax : axes, optional
            Axes object to plot data into
        debias : bool, optional
            If True, Any L or P data plotted will be debiased
        ratio : bool, optional
            If True, any t or f data will be converted to X/I and plotted
        ratio_rms_threshold, optional
            Mask Stokes ratios by ratio_rms_threshold * rms
        filename : str, optional
            filename to save figure to, by default None

        Returns
        -------
        fig : figure
            Return Figure Instance
        """        

        log_title(f"plotting [{data}] product.", col = "lblue")

        # get data
        pdat = self.get_data(data_list = data, get = True, debias = debias, 
                                ratio = ratio, ratio_rms_threshold=ratio_rms_threshold,
                                **kwargs)
        if not self._isdata():
            return None

        # plot 
        fig = plot_data(pdat, data, ax = ax, plot_type = self.plot_type)

        if self.save_plots:
            if filename is None:
                filename = f"{self.par.name}_{data}.png"
            plt.savefig(filename)

        if self.show_plots:
            plt.show()

        self._save_new_params()

        return fig




    # def zap_rfi(self, flagging_threshold = 10, rms_average = 1000, **kwargs):
    #     """
    #     Look for channels to zap based on RFI, algorithm developed by [Apurba Bera] and 
    #     revised (for speed) by [Tyson Dial]

    #     EXPERIMENTAL!!! - NEED TO TEST

    #     Take the terr_crop region and estimates the rfi there
    #     """
    #     log_title("Looking for RFI afflicted Channels to ZAP!!!", col = 'lblue')

    #     kwargs['tN'] = rms_average

    #     # initialise
    #     self._load_new_params(**kwargs)


    #     # get data
    #     data = self.get_data(["dsI", "fI"], get = True, t_crop = self.metapar.terr_crop,
    #                             **kwargs)

    #     if not self._iserr():
    #         log("Off-pulse crop required for RFI zappinh", lpf_col = self.pcol,
    #             stype = "err")
    #         raise ValueError("terr_crop undefined")

        
    #     # calculate channel rms and mask according to median and flag_threshold
    #     data['fI'] = np.nanstd(data['dsI'], axis = 1)
    #     med_rms = np.nanmedian(data['fI'])
    #     mad_rms = 1.48 * np.nanmedian(np.abs(data['fI'] - med_rms))

    #     # flag
    #     chanmask = np.ones(data['fI'].size, dtype = float)
    #     chan2flag = np.where(data['fI'] > (med_rms + flagging_threshold * mad_rms))[0]
    #     print(chan2flag)

    #     chanmask[chan2flag] = np.nan

    #     # convert to zap string
    #     zapstr = get_zapstr(chanmask, data['freq'])
    #     log(f"Channels to zap: {zapstr}", lpf_col = self.pcol)

    #     if (zapstr is None) or (zapstr == ""):
    #         return

    #     # add/set to zapchan
    #     if self.metapar.zapchan is None:
    #         self.metapar.zapchan = zapstr
    #     else:
    #         self.metapar.zapchan += f",{zapstr}"
        
    #     return
        





    # def zap_rfi(self, sigma = 3, **kwargs):
    #     """
    #     Zap channels based on if RFI is present, this assumes RFI will have an off-pulse RMS > sigma * RMS_median where 
    #     RMS_median is the median RMS of each channel. 

    #     Parameters
    #     ----------
    #     sigma : float, optional
    #         threshold for RFI zapping
        
    #     Returns
    #     -------
    #     zapchan : np.ndarray
    #         string of frequency channels to zap
        
    #     """
        
    #     log(f"zapping RFI", lpf_col = self.pcol)

    #     # get data
    #     data = self.get_data(["dsI", "fI"], get = True, **kwargs)

    #     if not self._iserr():
    #         log("Off-pulse crop required for RFI zappinh", lpf_col = self.pcol,
    #             stype = "err")
    #         return None
        
    #     # flag channels with RMS > then RMS threshold
    #     print(data['fIerr'])
    #     print(np.median(data['fIerr']))

    #     data['fIerr'] = np.nanstd(data['dsI'], axis = 1)
    #     med_rms = np.nanmedian(data['fIerr'])
    #     mad_rms = 1.48 * np.nanmedian(np.abs(data['fIerr'] - med_rms))
    #     data['dsI'][data['fIerr'] > (med_rms + sigma*mad_rms)] = np.nan



    #     # plot bad channels
    #     plt.figure(figsize = (10,10))
    #     plt.imshow(data['dsI'], aspect = 'auto', extent = [*self.this_par.t_lim, *self.this_par.f_lim])

    #     plt.show()







    def plot_stokes(self, ax = None, Ldebias = False, sigma = 2.0, 
            stk_type = "f", stk2plot = "IQUV", stk_ratio = False, filename: str = None, **kwargs):
        """
        Plot Stokes data, by default stokes I, Q, U and V data is plotted

        Parameters
        ----------
        ax: _axes_
            matplotlib.pyplot.axes object to plot to, default is None
        Ldebias : bool, optional
            Plot stokes L debias, by default False
        sigma : float, optional
            sigma threshold for error masking, data that is I < sigma * Ierr, mask it out or
            else weird overflow behavior might be present when calculating stokes ratios, by default 2.0
        stk_type : str, optional
            Type of stokes data to plot, "f" for Stokes Frequency data or "t" for time data, by default "f"
        stk2plot : str, optional
            string of stokes to plot, for example if "QV", only stokes Q and V are plotted, by default "IQUV"
        stk_ratio : bool, optional
            if true, plot stokes ratios S/I
        filename : str, optional
            name of file to save figure image, by default None
        **kwargs : Dict
            FRB parameter keywords

        Returns
        -------
        fig : figure
            Return figure instance

        """

        log_title(f"plotting stokes [{stk_type}] data", col = "lblue")

        # get data
        data_list = [f"{stk_type}I", f"{stk_type}Q", f"{stk_type}U", f"{stk_type}V"]
        data = self.get_data(data_list = data_list, get = True, **kwargs)
        if not self._isdata():
            return None

        err_flag = self._iserr()

        # check if off-pulse region given
        if not err_flag:
            log("Off-pulse crop required for plotting Ldebias", lpf_col = self.pcol,
                stype = "warn")
            Ldebias = False

        # data container for plotting
        pstk = {}

        if not stk_type in "ft":
            log("stk_type can only be t or f", lpf_col = self.pcol, stype = "err")

        # plot
        fig = plot_stokes(data, Ldebias = Ldebias, stk_type = stk_type,
                    sigma = sigma, stk2plot = stk2plot, stk_ratio = stk_ratio,
                    plot_type = self.plot_type, ax = ax) 

        
        if self.save_plots:
            if filename is None:
                filename = f"{self.par.name}_stk_{stk_type}.png"
            plt.savefig(filename)

        if self.show_plots:
            plt.show()
    

        self._save_new_params()

        return fig





    def plot_crop(self, stk = "I", filename = None,  **kwargs):
        """
        Plot current crop of of data along with off-pulse crop if given

        Parameters
        ----------
        stk : str, optional
            Stokes data to plot, by default "I"
        filename : str, optional
            name of file to save figure image, by default None
        """

        log_title("Plotting current crop parameters for visual inspection.", col = "lblue")

        # initialise
        self._load_new_params(**kwargs)



        # get crop in time and frequency, these will be used to draw the bounds of the crops in a larger
        # dynamic spectrum
        tcrop = self.this_metapar.t_crop.copy()
        fcrop = self.this_metapar.f_crop.copy()
        terrcrop = self.this_metapar.terr_crop.copy()

        # combine crops, these will be crops of the full dynamic spectrum  with bound markers
        fcrop_ds = [0.0, 1.0]       # by default take full bandwidth
        tpad = 50 / (self.par.t_lim[-1] - self.par.t_lim[0])   # by default we will pad time by 100ms

        # check of off-pulse region has been given
        err_flag = True
        if terrcrop is None:    # this essentially ignores the off-pulse region when plotting
            err_flag = False
            terrcrop = tcrop.copy()

        tcrop_ds = [0.0, 1.0]
        tcrop_ds[0] = min(tcrop[0], terrcrop[0]) - tpad
        tcrop_ds[1] = max(tcrop[1], terrcrop[1]) + tpad
        

        # cut crop to between [0.0, 1.0]
        if tcrop_ds[0] < 0.0:
            tcrop_ds[0] = 0.0
        if tcrop_ds[1] > 1.0:
            tcrop_ds[1] = 1.0



        if self.crop_units == "physical":
            tcrop_ds, fcrop_ds = self.par.phase2lim(t_crop = tcrop_ds,
                                                    f_crop = fcrop_ds)       
                                                    
        # get data
        kwargs['t_crop'] = [*tcrop_ds]
        kwargs['f_crop'] = [*fcrop_ds]

        self.get_data([f"ds{stk}"], **kwargs)



        # plot dynamic spectra
        fig = plt.figure(figsize = (12,12))
        plot_dynspec(self._ds[stk], aspect = 'auto', extent = [*self.this_par.t_lim, 
                                                             *self.this_par.f_lim])
        plt.xlabel("Time [ms]")
        plt.ylabel("Freq [MHz]")

        tcrop, fcrop = self.par.phase2lim(t_crop = tcrop, f_crop = fcrop)
        terrcrop, _ = self.par.phase2lim(t_crop = terrcrop)


        # plot on-pulse time region
        plt.plot([tcrop[0]]*2, self.this_par.f_lim, color = 'r', linestyle = "--", label = "On-pulse time crop")
        plt.plot([tcrop[1]]*2, self.this_par.f_lim, color = 'r', linestyle = "--")        
        
        # plot freq region
        plt.plot(self.this_par.t_lim, [fcrop[0]]*2, color = "orange", linestyle = "--", label = "freq crop")
        plt.plot(self.this_par.t_lim, [fcrop[1]]*2, color = "orange", linestyle = "--")

        # plot off-pulse time region
        if err_flag:
            plt.plot([terrcrop[0]]*2, self.this_par.f_lim, color = 'm', linestyle = "--", label = "Off-pulse time crop")
            plt.plot([terrcrop[1]]*2, self.this_par.f_lim, color = 'm', linestyle = "--")




        plt.legend()

        # title for crop info
        titstr = f"t crop: [{tcrop[0]:.1f}, {tcrop[1]:.1f}] [ms]\n"
        titstr += f"f crop: [{fcrop[0]:.1f}, {fcrop[1]:.1f}] [MHz]"
        if err_flag:
            titstr += f"\nterr crop: [{terrcrop[0]:.1f}, {terrcrop[1]:.1f}] [ms]"

        plt.title(titstr)

        if self.save_plots:
            if filename is None:
                filename = f"{self.par.name}_crop.png"
            plt.savefig(filename)

        if self.show_plots:
            plt.show()





    ## [ PLOT LORENTZ OF CROP ] ##
    def fit_scintband(self, method = "bayesian",priors: dict = None, statics: dict = None, 
                     fit_params: dict = None, redo = False, filename: str = None, n: int = None, **kwargs):
        """
        Fit for, Find and plot Scintillation bandwidth in FRB

        Parameters
        ----------
        method : str
            method for fitting \n
            [bayesian] - Use Bilby bayesian Statistics \n
            [least squares] - Use Scipy.Curve_fit least squares 
        priors : dict, optional
            Priors for sampling, by default None
        statics : dict, optional
            priors to keep constant, by default None
        fit_params : dict, optional
            extra arguments for Bilby.run_sampler function, by default None
        redo : bool, optional
            if True, will redo fitting in the case that results are cached, this is mainly for BILBY fitting, default is False
        filename : str, optional
            Save figure to file, by default None
        n : float, optional
            Polynomial order, by default None

        Returns
        -------
        p: pyfit.fit
            pyfit class structure
        """
        
        log_title(f"Fitting for Scintillation bandwidth using [{method}] method.", col = "lblue")
        ##====================##
        ##       get par      ##
        ##====================##

        # initilise dicts
        priors, statics, fit_params = dict_init(priors, statics, fit_params)

        # init pars
        self._load_new_params(**kwargs)
        


        ##====================##
        ##     do fitting     ##
        ##====================##

        # get data crop and spectrum
        self.get_data("fI", **kwargs)
        if not self._isdata():
            return None

        
        # in the case channel zapping has been performed, first calc residuals of non.nan values
        # then convert nans to zeros for acf.
        if self.zap:
            sumfunc = np.nansum
        else:
            sumfunc = np.sum
        
        # caculate acf of residuals
        if n is not None:
            y, yfit = residuals(self._f['I'], n = n)
            yrms = sumfunc(y**2)
        else:
            y = self._f['I']
            yrms = None
        y = acf(y)

        # in case zapping is involved
        mask = np.isnan(y)

        # lags
        x = np.linspace(self.this_par.df, self.this_par.bw - self.this_par.df,
                         y.size)


        # create instance of fitting
        yerr = None
        p = fit(x = x[~mask], y = y[~mask], yerr = None, func = lorentz, prior = priors,
                static = statics, fit_keywords = fit_params, method = method,
                residuals = self.residuals, plotPosterior = self.plotPosterior)

        # fit
        p.fit(redo = redo)

        # calculate modulation index
        # see (Macquart. j. P. et al, 2019) - [The spectral Properties of the bright FRB population]
        m = p.posterior['a'].val**0.5

        #  using error propogation and quick calculus to obtain error
        temp_err = (abs(p.posterior['a'].p) + abs(p.posterior['a'].m))/2 
        err = 0.5*temp_err/p.posterior['a'].val

        p.set_posterior('m', m, err, err)

        # set to fitted params
        self.fitted_params['scintband'] = p.get_posteriors()
        
        if self.verbose:
            p.stats()
        

            print(f"RMS in poly-n = {n} fitting (sum in square of residuals):")
            print(yrms)
        print(p)



        ##===================##
        ##   do plotting     ##
        ##===================##  
        if self.save_plots or self.show_plots:    

            if n is not None: 
                plt.figure(figsize = (10,10))
                plt.plot(self._freq, self._f['I'], 'k', label = "STOKES I spectra")
                plt.plot(self._freq, yfit(np.arange(self._f['I'].size)), 'r--', label = "STOKES I fit")
                plt.xlabel("Freq [MHz]")
                plt.ylabel("Flux (arb.)")
                plt.title(f"polyfit, n = {n}")
                plt.legend()

                if self.save_plots:
                    if filename is None:
                        filename = f"{self.par.name}_fit_scintband_broad_poly_model.png"
                    else:
                        filename += "_broad_poly_model.png"
                    
                    plt.savefig(filename)
                

            fig = p.plot(xlabel = "Freq [MHz]", ylabel = "Norm acf", show = False)

            if self.save_plots:
                if filename is None:
                    filename = f"{self.par.name}_fit_scintband.png"
                else:
                    filename += ".png"
                
                plt.savefig(filename)


            if self.show_plots:
                plt.show()


        # update instance par
        self._save_new_params()

        return p







    ## [ FIT SCATTERING TIMESCALE ] ##
    def fit_tscatt(self, method = "bayesian", npulse = 1, priors: dict = None, statics: dict = None, 
                   fit_params: dict = None, redo = False, filename: str = None, **kwargs):
        """
        Fit a series of gaussian's convolved with a one-sided exponential scattering tai using BILB

        Parameters
        ----------
        method : str
            method for fitting \n
            [bayesian] - Use Bilby bayesian Statistics \n
            [least squares] - Use Scipy.Curve_fit least squares 
        npulse : int, optional
            Number of gaussian to fit, by default 1
        priors : dict, optional
            Priors for sampling, by default None
        statics : dict, optional
            Priors to keep constant during fitting, by default None
        fit_params : dict, optional
            Keyword parameters for Bilby.run_sampler, by default None
        redo : bool, optional
            if True, will redo fitting in the case that results are cached, this is mainly for BILBY fitting, default is False
        filename : str, optional
            filename to save final plot to, by default None

        Returns
        -------
        p: pyfit.fit
            pyfit class structure
        """        
        log_title(f"Fitting for Scattering Time and overall burst profile using [{method}] method.", col = "lblue")
        ##====================##
        ## check if data valid##
        ##====================##

        # initilaise dicts
        priors, statics, fit_params = dict_init(priors, statics, fit_params)

        # init par
        self._load_new_params(**kwargs)


        ##====================##
        ##  proc data to fit  ##
        ##====================##
        
        # get data
        y = self.get_data("tI", get = True, **kwargs)["tI"]
        if not self._isdata():
            return None

        # create time profile
        x = np.linspace(*self.this_par.t_lim,y.size)

        err = None
        if self._iserr():
            err = self._t['Ierr']*np.ones(y.size)


        ##==================##
        ##   Do Fitting     ##
        ##==================##

        # create instance of fitting
        # the implemented convolution algorithm requires that we snap to integer samples
        # check if priors given or not
        p = fit(x = x, y = y, yerr = err, func = make_scatt_pulse_profile_func(npulse),
                prior = priors, static = statics, fit_keywords = fit_params, method = method,
                residuals = self.residuals, plotPosterior = self.plotPosterior) 

        # make sure we don't sample 0 for select priors
        if method == "least squares":
            for key in p.keys:
                if key == "sigma":
                    continue
                if key not in priors.keys():
                    print("KEYKEY")
                    if ("sig" in key) or ("tau" in key) or (key[0] == "a"):
                        p.bounds[key] = [0.0001, math.inf]

        # fit 
        p.fit(redo = redo)

        # set to fitted params
        self.fitted_params['tscatt'] = p.get_posteriors()
        self.fitted_params['tscatt']['npulse'] = npulse
        
        # print best fit parameters
        if self.verbose:
            p.stats()
        print(p)

        # plot
        if self.show_plots or self.save_plots:
            p.plot(xlabel = "Time [ms]", ylabel = "Flux Density (arb.)", show = False)

            if self.save_plots:
                if filename is None:
                    filename = f"{self.par.name}_fit_tscatt.png"
                
                plt.savefig(filename)
            
            if self.show_plots:
                plt.show()

        # save instance parameters
        self._save_new_params()

        return p

    












    def plot_poincare(self, stk_type = "f", sigma = 2.0, plot_data = True,
                        plot_model = False, n = 5, normalise = True, plot_1D_stokes = False, filename = None, **kwargs):
        """
        Plot Stokes data on a Poincare Sphere.

        Parameters
        ----------
        filename : str, optional
            filename to save figure to, by default None
        stk_type : str, optional
            types of stokes data to plot, by default "f" \n
            [f] - Plot as a function of frequency \n
            [t] - Plot as a function of time 
        sigma : float, optional
            Error threshold used for masking stokes data in the case that stokes/I is being calculated \n
            this avoids deviding by potentially small numbers and getting weird results,by default 2.0
        plot_data : bool, optional
            Plot Data on Poincare sphere, by default True
        plot_model : bool, optional
            Plot Polynomial fitted data on Poincare sphere, by default False
        normalise : bool, optional
            Plot data on surface of Poincare sphere (this will require normalising stokes data), by default True
        n : int, optional
            Maximum order of Polynomial fit, by default 5
        plot_1D_stokes: bool, optional
            if True, plot 1D stokes line plots seperately in another figure
        **kwargs : Dict
            FRB parameter keywords

        Returns
        -------
        fig : figure
            Return figure instance
        """    
        log_title(f"Plotting stokes [{stk_type}] data onto poincare sphere.", col = "lblue")

        self._load_new_params(**kwargs)

        if stk_type not in "tf":
            print("Stokes type must be either time (t) or frequency (f)")

        if not self._iserr():
            log("Must specify off-pulse crop region", stype = "err", lpf_col = self.pcol)
            return

        # get data
        data_list = []
        for S in "IQUV":
            data_list += [f"{stk_type}{S}"]

        self.get_data(data_list = data_list, **kwargs)
        if not self._isdata():
            return None

        # what type of data, time or freq
        if stk_type == "t":
            pdat = self._t
            cbar_lims = self.this_par.t_lim
            cbar_label = "Time [ms]"
        else:
            pdat = self._f
            cbar_lims = [self._freq[0], self._freq[-1]]
            cbar_label = "Frequency [MHz]"

        # plot poincare sphere
        fig, ax = create_poincare_sphere(cbar_lims = cbar_lims, cbar_label = cbar_label)

        stk_i, stk_m = plot_poincare_track(pdat, ax, sigma = sigma,
                    plot_data = plot_data, plot_model = plot_model, normalise = normalise,
                    n = n)
                    
        if self.save_plots:
            if filename is None:
                filename = f"{self.par.name}_poincare.png"
            else:
                filename += "_poincare.png"
            
            plt.savefig(filename)
        
        # also plot stokes params
        if plot_1D_stokes:
            if stk_type == "t":
                x = self._time
            else:
                x = self._freq
            x_m = np.linspace(*cbar_lims, stk_m['Q'].size)
            fig2, ax = plt.subplots(1, 1, figsize = (10,10))
            for S in "QUV":
                ax.plot(x, stk_i[S], label = S)
                ax.plot(x_m, stk_m[S], '--r')

            ax.set(xlabel = cbar_label, ylabel = "arb. ")
            ax.set_title("1D stokes plot")
            ax.legend()

            if self.save_plots:
                if filename is None:
                    filename = f"{self.par.name}_poincare_spectra_fit.png"
                else:
                    filename += "_poincare_spectra_fit.png"
                
                plt.savefig(filename)
            
        
        if self.show_plots:
            plt.show()

        self._save_new_params()

        return fig, fig2













    def fit_RM(self, method = "RMquad", sigma: float = None, rm_prior: list = [-1000, 1000], 
                pa0_prior: list = [-np.pi/2, np.pi/2], fit_params: dict = None, filename: str = None, **kwargs):
        """
        Fit Spectra for Rotation Measure

        Parameters
        ----------
        method : str, optional
            Method to perform Rotation Measure fitting, by default "RMquad" \n
            [RMquad] - Fit for the Rotation Measure using the standard quadratic method \n
            [RMsynth] - Use the RM-tools RM-Synthesis method \n
            [QUfit] - Fit log-likelihood model of Stokes Q and U parameters (see bannister et al 2019 - supplementary)
        sigma : float, optional
            Apply masking based on S/N threshold given, used in [RMquad, RMsynth, QUfit]
        rm_prior : list
            priors for rotation measure, used in [QUfit], by default [-1000, 1000]
        pa0_prior : list
            priors for PA0, used in [QUfit], by default [-pi/2, pi/2] (Shouldn't need to change)
        fit_params : dict, optional
            keyword parameters for fitting method, by default None \n
            [RMquad] - Scipy.optimise.curve_fit keyword params \n
            [RMsynth] - RMtools_1D.run_synth keyword params \n
            [QUfit] - bilby.run_sampler keyword params
        filename : str, optional
            filename to save figure to, by default None

        Returns
        -------
        p : pyfit.fit
            pyfit class fitting structure
        """
        log_title(f"Fitting for RM using [{method}] method.", col = "lblue")

        fit_params = dict_init(fit_params)
        self._load_new_params(**kwargs)

        # check which data products are needed
        if method in ["RMsynth", "RMquad", "QUfit"]:
            data_list = ["fI", "fQ", "fU"]
        
        else:
            log("Invalid method for estimating RM", stype = "err", lpf_col = self.pcol)
            return None
            
        if self.this_metapar.terr_crop is None:
            log("Must specify 'terr_crop' for rms crop if you want to use RMsynth or RMquad", stype = "err",
                lpf_col = self.pcol)
            return (None, ) * 5
        
        ## get data ##
        self.get_data(data_list, ignore_nans = True, **kwargs)
        if not self._isdata():
            return None

        ## mask data based on S/N threshold given
        if sigma is not None:
            mask = self._f["I"] > self._f['Ierr'] * sigma
        else:
            mask = np.ones(self._f['I'].size, dtype = bool)

        ## run fitting for RM ##
        if method == "RMquad":
            # run quadrature method
            if self.this_par.f0 is None:
                log("f0 not given, using middle of band", stype = "warn", lpf_col = self.pcol)
                self.this_par.f0 = self.this_par.cfreq

            f0 = self.this_par.f0
            # run fitting
            rm, rm_err, pa0, pa0_err = fit_RMquad(self._f['Q'][mask], self._f['U'][mask], self._f['Qerr'][mask],
                                                  self._f['Uerr'][mask], self._freq[mask], f0, **fit_params)


        elif method == "RMsynth":
            # run RM synthesis method
            pa0_err = 0.0       # do this for now, TODO
            I, Q, U = self._f['I'], self._f['Q'], self._f['U']
            Ierr, Qerr, Uerr = self._f['Ierr'], self._f['Qerr'], self._f['Uerr'] 
            rm, rm_err, f0, pa0 = fit_RMsynth(I[mask], Q[mask], U[mask], Ierr[mask], 
                                Qerr[mask], Uerr[mask], self._freq[mask], **fit_params)

        
        elif method == "QUfit":

            # TODO: make reference frequency same as FDF?

            # run log-likelihood estimating for Q and U paramters
            f0 = 0.0
            Q, U = self._f['Q'], self._f['U']
            Ierr, Qerr, Uerr = self._f['Ierr'], self._f['Qerr'], self._f['Uerr']
            rm, rm_err, pa0, pa0_err = RM_QUfit(Q = Q[mask], U = U[mask], Ierr = Ierr[mask], Qerr = Qerr[mask], 
                                                Uerr = Uerr[mask], f = self._freq[mask], rm_priors = rm_prior, 
                                                pa0_priors = pa0_prior, **fit_params)


        # function for plotting diagnostics
        if method == "QUfit":
            def rmquad(f, rm, pa0):
                angs = pa0 + rm*c**2/(f*1e6)**2
                return 90/np.pi*np.arctan2(np.sin(2*angs), np.cos(2*angs))
        else:
            def rmquad(f, rm, pa0):
                angs = pa0 + rm*c**2/1e12*(1/f**2 - 1/f0**2)
                return 90/np.pi*np.arctan2(np.sin(2*angs), np.cos(2*angs))


        # put into pyfit structure
        PA, PA_err = calc_PA(self._f['Q'][mask], self._f['U'][mask], self._f['Qerr'][mask], self._f['Uerr'][mask])

        p = fit(x = self._freq[mask], y = 180/np.pi*PA, yerr = 180/np.pi*PA_err, func = rmquad,
                 residuals = self.residuals)
        p.set_posterior('rm', rm, rm_err, rm_err)
        p.set_posterior('pa0', pa0, pa0_err, pa0_err)
        p.set_posterior('f0', f0, 0.0, 0.0)

        # set values to fitted_params 
        self.fitted_params['RM'] = p.get_posteriors()
        self.fitted_params['RM']['f0'] = _posterior(f0, 0, 0)
        p._is_fit = True
        p._is_stats = True
        p._get_stats()
        print(p)

        # plot
        if self.save_plots or self.show_plots:
            
            p.plot(xlabel = "Frequency [MHz]", ylabel = "PA [deg]", ylim = [-90, 90], show = False)

            if self.save_plots:
                if filename is None:
                    filename = f"{self.par.name}_RM_fit.png"
                
                plt.savefig(filename)

            if self.show_plots:
                plt.show()


        self._save_new_params()


        return p








    




    def plot_PA(self, Ldebias_threshold = 2.0, stk2plot = "ILV", flipPA = False, stk_ratio = False,
                fit_params: dict = None, filename: str = None, save_files = False, **kwargs):
        """
        Plot Figure with PA profile, Stokes Time series data, and Stokes I dyspec. If RM is not 
        specified, will be fitted first.

        Parameters
        ----------
        Ldebias_threshold : float, optional
            Sigma threshold for PA masking, by default 2.0
        stk2plot : str, optional
            string of stokes to plot, for example if "QV", only stokes Q and V are plotted, \n
            by default "IQUV", choice between "IQUVLP"
        flipPA : bool, optional
            Plot PA between [0, 180] degrees instead of [-90, 90], by default False
        stk_ratio: bool, optional
            Plot Stokes ratios in time series ax, by default False
        fit_params : dict, optional
            keyword parameters for fitting method, by default None \n
            [RMquad] - Scipy.optimise.curve_fit keyword params \n
            [RMsynth] - RMtools_1D.run_synth keyword params
        filename : str, optional
            filename of figure to save to, by default None
        save_files : Bool, optional
            if true, will save 1D .npy file with PA and .npy file with PAerrs, by default False

        Returns
        -------
        fig : figure
            Return figure instance
        """
        log_title("Plotting PA mosaic image with PA profile, Stokes profile and Dynamic spectra.", col = "lblue")

        # initialise parameters
        fit_params = dict_init(fit_params)

        self._load_new_params(**kwargs)

        if self.this_metapar.terr_crop is None:
            log("Need to specify 'terr_crop' for rms estimation", stype = "err", lpf_col = self.pcol)
            return None

        if self.this_par.RM is None:
            log("RM not specified, either provide an RM or fit for one using .fit_RM()", lpf_col = self.pcol, stype = "err")
            return None

        
        ## get data
        data_list = ["dsI", "dsQ", "dsU", 
                       "tI",  "tQ",  "tU", 
                       "fQ",  "fU", "tV"]
        self.get_data(data_list, **kwargs)
        if not self._isdata():
            return None

        ## calculate PA
        stk_data = {"tQ":self._t["Q"], "tU":self._t["U"], "tQerr":self._t["Qerr"],
                    "tUerr":self._t["Uerr"], "tIerr":self._t["Ierr"]}
        PA, PA_err = calc_PAdebiased(stk_data, Ldebias_threshold = Ldebias_threshold)

        # create figure
        fig, AX = plt.subplot_mosaic("P;S;D", figsize = (12, 10), 
                    gridspec_kw={"height_ratios": [1, 2, 2]}, sharex=True)

        _x = np.linspace(*self.this_par.t_lim, PA.size)

        ## plot PA
        plot_PA(_x, PA, PA_err, ax = AX['P'], flipPA = flipPA)

        ## plot Spectra
        pdat = {'time':_x}
        for S in "IQUV":
            pdat[f"t{S}"] = self._t[S]
            pdat[f"t{S}err"] = self._t[f"{S}err"]

        plot_stokes(pdat, ax = AX['S'], stk_type = "t", stk2plot = stk2plot, Ldebias = True, 
                    plot_type = self.plot_type, stk_ratio = stk_ratio)

        ## plot dynamic spectra
        ds_freq_lims = fix_ds_freq_lims(self.this_par.f_lim, self.this_par.df)
        plot_dynspec(self._ds['I'], ax = AX['D'], aspect = 'auto', 
                       extent = [*self.this_par.t_lim,*ds_freq_lims])
        AX['D'].set_ylabel("Frequency [MHz]", fontsize = 12)
        AX['D'].set_xlabel("Time [ms]", fontsize = 12)

        # adjust figure
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0)
        AX['P'].get_xaxis().set_visible(False)
        AX['S'].get_xaxis().set_visible(False)

        if self.save_plots:
            if filename is None:
                filename = f"{self.par.name}_PA_mosaic.png"
            
            plt.savefig(filename)

        if save_files:
            print(f"Saving PA data to {self.par.name}_PA.npy...")
            np.save(f"{self.par.name}_PA.npy", PA)

            print(f"Saving PA err data to {self.par.name}_PAerr.npy...")
            np.save(f"{self.par.name}_PAerr.npy", PA_err)

        
        if self.show_plots:
            plt.show()


        self._save_new_params()


        return fig
                                                



    def calc_polfracs(self, debias = False, peak_sigma = 3.0, peak_average_factor = 1, **kwargs):
        """
        Calculate polarisation fractions using a number of different methods.

        Parameters
        ----------
        debias : bool, optional
            Debiases Stokes L, P and |V|, |Q| and |U|, by default False.
        peak_sigma : float, optional
            Provide a threshold in terms of I/Ierr that will be used to mask the data
            before estimating the peak fraction of each stokes parameter. This will be 
            nessesary to filter out noisy data.
        peak_average_factor, float
            averaging (downsampling) factor to apply to X(t) stokes profiles to help estimate their peaks
        """

        log_title("Calculating Polarisation fractions", col = "lblue")

        # We wont over complicate this, only calculate polarisation fractions if
        # all stokes dynamic spectra are loaded
        loaded_stk = ""
        for s in "IQUV":
            if self.ds[s] is None:
                log(f"Stokes {s} dynamic spectrum is missing, make sure all Stokes dynspecs are loaded in!", stype = "err")
                return

        # proc KWARGS
        self._load_new_params(**kwargs) 

        # get data
        S = self.get_data(["tI", "tQ", "tU", "tV", "tL", "tP"], get = True, debias = debias, **kwargs)
        nsamp = S['tI'].size

        # check if error was given, else turn off debias
        err = True
        if self.this_metapar.terr_crop is None:
            debias = False
            err = False
            log("No off-pulse crop given to calculate dibased L, P and/or |U/Q/V|, specify [terr_crop]...", stype = "warn")
            log("No peak fractions we be calculatedm specify [terr_crop]...", stype = "warn")

        
        # calculated integrated stokes
        intS = {}
        for s in "IQUVLP":
            intS[s] = np.nansum(S[f't{s}'])
            if err:
                if s in "LP":
                    intS[f'{s}err'] = np.nansum(S[f't{s}err']**2)**0.5
                else:
                    intS[f'{s}err'] = nsamp**0.5 * S[f't{s}err']
            else:
                intS[f"{s}err"] = None

        # calculate integrated absolute Stokes Q, U and V
        for s in "QUV":
            if err:
                intS[f"abs{s}"] = np.nansum(calc_stokes_abs_debias(S[f't{s}'], S['tIerr']))
            else:
                intS[f"abs{s}"] = np.nansum(np.abs(S[f't{s}']))
        
        # calculate Stokes fractions
        fracS = {}
        polname = ["I", "q", "u", "v", "l", "p"]
        for i, s in enumerate("IQUVLP"):
            fracS[polname[i]], fracS[f'{polname[i]}err'] = calc_ratio(intS['I'], intS[s], 
                                                intS['Ierr'], intS[f'{s}err'])
        polname = ["|q|", "|u|", "|v|"]
        for i, s in enumerate("QUV"): # absolute values
            fracS[polname[i]], fracS[f'{polname[i]}err'] = calc_ratio(intS['I'], intS[f'abs{s}'],
                                                intS['Ierr'], intS[f'{s}err'])



        # get peak Q, U, V, L and P
        
        if err:
            # average Stokes I for masking
            peaks = {}
            peaks_pos = {}

            # find mask based on sigma value
            mask = (average(S['tI'], N = peak_average_factor, nan = True)
                    /average(S['tIerr'], N = peak_average_factor, nan = True)) < peak_sigma
            stk_frac = {}
            S['time'] = average(S['time'], N = peak_average_factor)

            print(S['tI'].size)
            print(S['tQ'].size)

            for s in "QUVLP":
                # get Stokes ratio
                stk_frac[s.lower()], stk_frac[f'{s.lower()}err'] = calc_ratio(S['tI'], S[f't{s}'], S['tIerr'], S[f't{s}err'])

                # average
                stk_frac[s.lower()] = average(stk_frac[s.lower()], N = peak_average_factor, nan = True)
                stk_frac[f'{s.lower()}err'] = average(stk_frac[f'{s.lower()}err'], N = peak_average_factor, nan = True)

                # mask
                stk_frac[s.lower()][mask] = np.nan
                stk_frac[f'{s.lower()}err'][mask] = np.nan
                
                # find peak
                peak_ind = np.nanargmax(np.abs(stk_frac[s.lower()]))
                peaks[s.lower()] = stk_frac[s.lower()][peak_ind]
                peaks[f"{s.lower()}err"] = stk_frac[f'{s.lower()}err'][peak_ind]
                peaks_pos[s.lower()] = S['time'][peak_ind]
            
            # diagnostic plots
            fig, ax = plt.subplots(1, 1, figsize = (12,8))
            
            for s in "quvlp":
                _PLOT(S['time'], stk_frac[s], stk_frac[f'{s}err'], ax = ax, plot_type = self.plot_type, 
                        color = _G.stk_colors[s.upper()])
            ylim = ax.get_ylim()
            for s in "quvlp":
                # plot marker
                ax.plot([peaks_pos[s]]*2, ylim, color = _G.stk_colors[s.upper()], linestyle = "--",
                            label = f'${s}_{{peak}}$ at t = {peaks_pos[s]:.2f} ms')
            
            ax.set(ylim = ylim, xlabel = "Time [ms]", ylabel = "Stokes X/I fraction")
            ax.legend()

            # save figure
            if self.save_plots:
                filename = f"{self.par.name}_peak_polfracs.png"
            
                plt.savefig(filename)



        # Now we can print everything out
        debias_flag = "FALSE\n"
        if debias:
            debias_flag = "TRUE\n"
        print("\nStokes fractions:")
        print("="*50)
        print(f"debiased = ", debias_flag)

        def _print_err(val):
            if err:
                return f"{val:.4f}"
            else:
                return "None"


        print("=======  CONTINUUM-ADDED Stokes fractions  =======\n")
        print("These fractions are calculated by first")
        print("integrating over the debiased polarisation profile")
        print("="*50, "\n")
        print("Legend:")
        print("l = sum(L(t))/sum(I(t))\n")
        print("|q|".ljust(15), f"{fracS['|q|']:.4f} +/- {_print_err(fracS['|q|err'])}")
        print("|u|".ljust(15), f"{fracS['|u|']:.4f} +/- {_print_err(fracS['|u|err'])}")
        print("|v|".ljust(15), f"{fracS['|v|']:.4f} +/- {_print_err(fracS['|v|err'])}")
        print("l".ljust(15), f"{fracS['l']:.4f} +/- {_print_err(fracS['lerr'])}")
        print("p".ljust(15), f"{fracS['p']:.4f} +/- {_print_err(fracS['perr'])}\n")
        
        print("Integrated (Signed) Stokes Paramters")
        print("q".ljust(15), f"{fracS['q']:.4f}".ljust(7) + f" +/- {_print_err(fracS['qerr'])}")
        print("u".ljust(15), f"{fracS['u']:.4f}".ljust(7) + f" +/- {_print_err(fracS['uerr'])}")
        print("v".ljust(15), f"{fracS['v']:.4f}".ljust(7) + f" +/- {_print_err(fracS['verr'])}\n")

        print("====  Vector-addded Stokes L and P fractions  ====")
        print("="*50, "\n")

        print("Legend:")
        print("l* = sqrt(q^2 + u^2)")
        print("|l|* = sqrt(|q|^2 + |u|^2)")
        print("p* = sqrt(q^2 + u^2 + v^2)")
        print("|p|* = sqrt(|q|^2 + |u|^2 + |v|^2)\n")


        # calculate l* and p*
        fracS['l*'], fracS['l*err'] = calc_L(fracS['q'], fracS['u'], fracS['qerr'], fracS['uerr'])
        fracS['p*'], fracS['p*err'] = calc_P(fracS['q'], fracS['u'], fracS['v'], fracS['qerr'],
                                    fracS['uerr'], fracS['verr'])
        print("l*".ljust(15), f"{fracS['l*']:.4f} +/- {_print_err(fracS['l*err'])}")
        print("p*".ljust(15), f"{fracS['p*']:.4f} +/- {_print_err(fracS['p*err'])}")

        # calculate |l|* and |p|*
        fracS['|l|*'], fracS['|l|*err'] = calc_L(fracS['|q|'], fracS['|u|'], fracS['|q|err'],
                                        fracS['|u|err'])
        fracS['|p|*'], fracS['|p|*err'] = calc_P(fracS['|q|'], fracS['|u|'], fracS['|v|'],
                                        fracS['|q|err'], fracS['|u|err'], fracS['|v|err'])
        print("|l|*".ljust(15), f"{fracS['|l|*']:.4f} +/- {_print_err(fracS['|l|*err'])}")
        print("|p|*".ljust(15), f"{fracS['|p|*']:.4f} +/- {_print_err(fracS['|p|*err'])}\n")

        print("= Total polarisation fraction calculated L and V =")
        print("="*50, "\n")
        print("Legend:")
        print("p^ = sqrt(l^2 + v^2)")
        print("|p|^ = sqrt(l^2 + |v|^2)\n")

        fracS['p^'], fracS['p^err'] = calc_L(fracS['l'], fracS['v'], fracS['lerr'], fracS['verr'])
        fracS['|p|^'], fracS['|p|^err'] = calc_L(fracS['l'], fracS['|v|'], fracS['lerr'], fracS['|v|err'])
        print("p^".ljust(15), f"{fracS['p^']:.4f} +/- {_print_err(fracS['p^err'])}")
        print("|p|^".ljust(15), f"{fracS['|p|^']:.4f} +/- {_print_err(fracS['|p|^err'])}\n")

        if err:
            print(f"= Peak absolute polarisation fraction at dt = [{self.this_par.dt * 1000:.0f}] us =")
            print(f"="*50, "\n")
            for s in "quvlp":
                print(f"{s}_peak".ljust(15), f"{peaks[s]:.4f} +/- {peaks[f'{s}err']:.4f}".ljust(25), f"at time t = {peaks_pos[s]:.2f} ms")
            if self.save_plots:
                print(f"\nPrinting out diagnostic plot of stokes polarisation fractions as a function of time [{filename}]\n")

        if self.show_plots and err:
            plt.show()
        

        return fracS








