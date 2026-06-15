##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 30/04/2026 
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
import math
import os, sys
from copy import deepcopy
import inspect
from .pyfit import fit, _posterior
import yaml
from .frbutils import set_dynspec_plot_properties
from .utilmethods import *

## import utils ##
from .utils import (dict_get, dict_edit_and_copy,
                    dict_init, dict_isall,
                    merge_dicts, dict_null, get_stk_from_datalist, 
                    set_plotstyle, fix_ds_freq_lims, sort_legend)
from .io import ilexIO, load_data, save_data

from .data import *

## import FRB stats ##
from .fitting import (fit_RMquad, fit_RMsynth, RM_QUfit, lorentz, lorentz_yshifted,
                     make_scatt_pulse_profile_func, tscattLikelihood ,scatt_pulse_profile,
                     scatt_pulse_profile_relative, burnslaw)

## import FRB params ##
from .par import FRB_params, FRB_metaparams

# ## import FRB htr functions ##
# from .htr import make_stokes

## import globals ##
from .globals import _G, c

## import plot functions ##
from .plot import (plot_RM, plot_PA, plot_stokes,      
                  plot_poincare_track, create_poincare_sphere, plot_data, _PLOT, plot_dynspec, plot)

## import processing functions ##
from .logging import log, get_verbose, set_verbose, log_title, strcol
from .master_proc import master_proc_data
from .widths import *

# interactive module
from .interactive import ZapInteractive, medrms_chanflag


    

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
    show_dynzaps: bool
        If true, show zapped channels in dynspec as Patches
    plot_tpad: float
        Additional width in [ms] to pad in time when plotting data, this is only for visual purposes and will not affect processing, by default 30.0 
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
    IOconfig: ilex.io.ilexIO
        ilex IO instance

    """




    ## [ INITIALISE FRB ] ##
    def __init__(self, yaml_file = None, name: str = _G.p['name'],    RA: str = _G.p['RA'],    DEC: str = _G.p['DEC'], 
                       MJD: float = _G.p['MJD'], DM: float = _G.p['DM'],      bw: int = _G.p['bw'],    cfreq: float = _G.p['cfreq'], 
                       t_crop = None,               f_crop = None,           tN: int = 1,
                       fN: int = 1,                 t_lim_base = _G.p['t_lim_base'],   f_lim_base = _G.p['f_lim_base'],
                       RM: float = _G.p['RM'],      f0: float = _G.p['f0'],  pa0: float = _G.p['pa0'],
                       verbose: bool = _G.hp['verbose'], norm = _G.mp['norm'], dt: float = _G.p['dt'], 
                       df: float = _G.p['df'],      zapchan: str = _G.mp['zapchan'], terr_crop = None, t_ref = _G.p['t_ref'],      
                       plot_tpad: float = _G.hp['plot_tpad']):
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
                        terr_crop = terr_crop, tN = tN, fN = fN, norm = norm, zapchan = zapchan)


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
        self.plot_tpad = plot_tpad
        self.show_dynzaps = True
        self.crop_units = "physical"
        self.zap = False                # if True, will treat arrays as zapped

        # weightings
        self.apply_tW = True                  # apply time weights
        self.apply_fW = True                  # apply freq weights

        self._isinstance = False        # if data instance is valid
        self.fitted_params = {}

        # plotting stuff
        self.dynspec_cmap = "viridis"
        self.dynspec_satlvl = 0
        self.dynspec_cnorm = "linear"
        self.dynspec_cmap_alpha = 0.5
        self.mplstyle = None
        self.dynspec_interp = 'none'

        # IO 
        self.ilexIO = ilexIO(frb = self)


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


    @property
    def dynspec_satlvl(self):
        return self._dynspec_satlvl
    
    @dynspec_satlvl.setter
    def dynspec_satlvl(self, satlvl):

        self._dynspec_satlvl = satlvl

        set_dynspec_plot_properties(satlvl = satlvl)


    @property
    def dynspec_cnorm(self):
        return self._dynspec_cnorm

    @dynspec_cnorm.setter
    def dynspec_cnorm(self, cnorm):

        if cnorm not in ["linear", "exp", "power"]:
            print(f"cnorm: {cnorm} not valid, setting to 'linear'...")
            cnorm = 'linear'

        self._dynspec_cnorm = cnorm

        set_dynspec_plot_properties(cnorm = cnorm)


    @property
    def dynspec_cmap_alpha(self):
        return self._dynspec_cmap_alpha
    
    @dynspec_cmap_alpha.setter
    def dynspec_cmap_alpha(self, cmap_alpha):

        self._dynspec_cmap_alpha = cmap_alpha

        set_dynspec_plot_properties(cmap_alpha = cmap_alpha)

    @property 
    def dynspec_interp(self):
        return self._dynspec_interp

    @dynspec_interp.setter
    def dynspec_interp(self, interp):
        
        self._dynspec_interp = interp

        set_dynspec_plot_properties(interpolation = interp)

    @property
    def mplstyle(self):
        return self._mplstyle
    
    @mplstyle.setter
    def mplstyle(self, mplstyle):

        self._mplstyle = mplstyle

        if mplstyle is None:    # set as default
            self._mplstyle = os.path.join(os.environ['ILEX_PATH'], 
                                          "files/default.mplstyle")
        print(self._mplstyle)
        plt.style.use(self._mplstyle)



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
            self.ilexIO.set(filepath = yaml_file)
            yaml_pars = self.ilexIO.load_pars()
            # yaml_pars = load_param_file(yaml_file)

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
            # set_plotstyle(yaml_pars['plots']['plotstyle_file'])
            # if yaml_pars['plots']['plotstyle_file'] is None:
            #     log("Setting plotting style: Default")
            # else:
            #     log(f"setting plotting style: {yaml_pars['plots']['plotstyle_file']}")


        def init_par_from_load(x):
            """
            Initialise a number of parameters from loaded file
            """

            self.par.nchan = x.shape[0]                     # assumed that dyn spec is [freq,time]
            self.par.nsamp = x.shape[1]    
            self.par.t_lim_base  = [0.0, self.par.dt * self.par.nsamp]


        ## dict. of files that will be loaded in
        data_files = {"dsI": dsI, "dsQ": dsQ, "dsU": dsU, "dsV": dsV}
        for dkey in data_files.keys():
            if data_files[dkey] is not None:
                data_files[dkey] = os.path.abspath(data_files[dkey])
        self._data_files = deepcopy(data_files)
        old_chans = None

        # loop through files
        load_zapchan = ""

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
                    load_zapchan = get_zapstr(chans, self.par.get_freqs())
                    if old_chans is not None:
                        if not np.all(old_chans == chans):
                            log("Channels being zapped are different for each Stokes Dynamic spectra!!", stype = "warn")
                    old_chans = chans.copy()
        self.metapar.zapchan = combine_zapchan(self.metapar.zapchan, load_zapchan)








        

        
    ## [ SAVING FUNCTION - SAVE CROP OF DATA ] ##
    def save_data(self, data_list = None, name = None, save_yaml = False, yaml_file = None, stk_debias = False, stk_ratio = False,
                 proc = False, overwrite = False, **kwargs):
        """
        Save current instance data

        Parameters
        ----------
        data_list : List(str), optional
            List of data to save, by default None
        name : str, optional
            Common Pre-fix for saved data, by default None, if None the name parameter of the
            FRB class will be used.        
        stk_debias: bool
            Debias Stokes data before saving
        stk_ratio: bool
            Save stokes ratios
        proc: bool
            save copy of processed data to new config file, by default False
        overwrite: bool
            Overwrite yaml file, by default False
        """

        log_title("Saving Stokes data, the data is saved as .npy files. ", col = "lblue")

        if (yaml_file is None) and (not proc):
            overwrite = True

        if save_yaml:
            log("Saving fitted parameters to yaml file...", lpf_col = "green")
            self.ilexIO.set(proc = proc,
                           overwrite = overwrite,
                           filepath = yaml_file)
            self.ilexIO.save()
            # save_frb_to_param_file(self, yaml_file)


        if data_list is None:
            log("No data specified for saving...", stype = "warn")
            return


        print("Saving the following data products:")
        for data in data_list:
            print(f"[{data}]")

        # get data
        pdat = self.get_data(data_list, stk_debias = stk_debias, stk_ratio = stk_ratio, get = True)
        if not self._isdata():
            return 
        

        if name is None:
            frbname = str(self.par.name)
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
    def get_data(self, data_list = "dsI", get = False, ignore_nans = False, stk_debias = False, 
                    stk_ratio = False, stk_sigma = None, **kwargs):
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
        stk_debias, bool, optional
            If true, tL/fL and tP/fP will be debiased 
        stk_ratio, bool, optional
            If true, calculate X/I for t and f products
        stk_sigma : float, optional
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
        data_products = self._init_proc(data_list, stk_debias = stk_debias, stk_ratio = stk_ratio)

        ## first check if there is data to use
        if not self._isvalid(data_products):
            log("Loaded data not avaliable or incorrect DS shapes", stype = "err",
                lpf_col = self.pcol)
            self._isinstance = False
            return 

        ## make new instances
        self._make_instance(data_list = data_list, ignore_nans = ignore_nans, stk_debias = stk_debias, stk_ratio = stk_ratio,
                            stk_sigma = stk_sigma)


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



    def _make_instance(self, data_list = None, ignore_nans = False, stk_debias = False, stk_ratio = False,
                        stk_sigma = None):
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
                                            data_list, full_par, stk_debias, stk_ratio, stk_sigma)

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



    
    def _init_proc(self, data_list, stk_debias = False, stk_ratio = False):
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
                if stk_debias:
                    add_stokes_I = True
        if add_stokes_I:
            log("Added stokes I to process for debiasing L and/or P polarisations", lpf = False)
            if "I" not in stk:
                stk += "I"
        
        # if calculating ratios
        if stk_ratio:
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

        # if self.crop_units not in ["physical", "phase"]:
        #     log("Units for cropping must be one of: ['physical', 'phase'] ", stype = "err")
        #     return
        
        if self.crop_units != "physical":
            self.crop_units = "physical"
            log("Only 'physical' crop units, i.e. [ms and MHz] are supported now...", stype = "warn")


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


    def get_crop(self, units = "physical", **kwargs):
        """
        Auxillary function to return tcrop and fcrop in desired units 

        Parameters
        ----------
        units: str
            physical -> returns crop in units of ms and MHz for tcrop and fcrop \n
            phase -> returns crop in units of phase of full array in time and freq \n
            index -> returns crops in index format 
        **kwargs

        Returns
        -------
        tcrop : list[float]
            on-pulse time crop
        fcrop : list[float]
            freq crop
        terr_crop : list[float]
            off-pulse time crop
        """

        if units not in ["physical", "phase", "index"]:
            print("Units type is invalid for get_crop")
            return

        self._load_new_params(**kwargs)
        terr_crop = None
        
        if units == "phase":
            t_crop, f_crop = self.this_metapar.t_crop.copy(), self.this_metapar.f_crop.copy()
            if self._iserr():
                terr_crop = self.this_metapar.terr_crop.copy()
        
        elif units == "physical":
            t_crop, f_crop = self.par.phase2lim(t_crop = self.this_metapar.t_crop,
                                           f_crop = self.this_metapar.f_crop)
            if self._iserr():
                terr_crop, _ = self.par.phase2lim(t_crop = self.this_metapar.terr_crop)
        
        else:
            t_crop = self.this_metapar.t_crop.copy()
            f_crop = self.this_metapar.f_crop.copy()
            t_crop = [int(t_crop[0]*self.this_par.nsamp), int(t_crop[1]*self.this_par.nsamp)]
            f_crop = [int(f_crop[0]*self.this_par.nchan), int(f_crop[1]*self.this_par.nchan)]
            if self._iserr():
                terr_crop = self.thispar.terr_crop.copy()
                terr_crop = [int(terr_crop[0]*self.this_par.nsamp), int(terr_crop[1]*self.this_par.nsamp)]
        print(t_crop, terr_crop, self.this_metapar.t_crop)
        return t_crop, f_crop, terr_crop


    def get_ptcrop(self, tcrop = None):

        # get plot crop, this is t_crop with plot_tpad
        if tcrop is None:
            t_crop = self.metapar.t_crop.copy()
        else:
           t_crop = tcrop.copy() 
        if type(t_crop[0]) == str:
            if t_crop[0] == "min":
                t_crop[0] = self.par.t_lim[0]
            else:
                ValueError(f"t_crop[0] = {t_crop[0]} incorrect! either provide a value or use 'min'...")
        if type(t_crop[1]) == str:
            if t_crop[1] == "max":
                t_crop[1] = self.par.t_lim[1]
            else:
                ValueError(f"t_crop[1] = {t_crop[1]} incorrect! either provide a value or use 'max'...")

        if self.plot_tpad <= 0.0:
            return t_crop
        
        pt_crop =  [t_crop[0] - self.plot_tpad/2,
                t_crop[1] + self.plot_tpad/2]

        # check bounds
        if pt_crop[0] < self.par.t_lim[0]:
            pt_crop[0] = self.par.t_lim[0]
        if pt_crop[1] > self.par.t_lim[1]:
            pt_crop[1] = self.par.t_lim[1]

        return pt_crop.copy()



    def set_zeropoint(self, x = 0.0, method = "val", **kwargs):

        if method not in ["val", "max", "centroid"]:
            print("Method for setting zeropoint in time must be either 'val', 'max' or 'centroid'")
            return
        
        if method == "val":
            if type(t_ref) != float:
                print("t_ref must be a floating point using method = 'val'")
                return
            t_ref = x
        
        elif method == "max":
            self._load_new_params(**kwargs)
            data = self.get_data('tI', get = True, **kwargs)
            t_ref = data['time'][np.argmax(data['tI'])]
        
        else:
            self._load_new_params(**kwargs)
            data = self.get_data('tI', get = True, **kwargs)
            t_ref = get_centroid(data['time'], data['tI'])


        # get offset time based crops
        t_crop, _, terr_crop = self.get_crop()
        t_crop = [t_crop[0] - t_ref,t_crop[1] - t_ref]
        if self._iserr():
            terr_crop = [terr_crop[0] - t_ref, terr_crop[1] - t_ref]

        # set new crops 
        self.set(t_ref = t_ref, t_crop = t_crop, terr_crop = terr_crop)
    
        return




    ##===============================================##
    ##             Diagnostic functions              ##
    ##===============================================##


    def __str__(self):
        """
        Info:
            Print info about FRB class

        """

        strpad = 20

        def center_str(_str, totlen):
            strlen = len(_str)
            remlen = totlen - strlen
            if remlen <= 0:
                return _str
            lrem = remlen // 2
            rrem = remlen - lrem
            return "".ljust(lrem) + _str + "".ljust(rrem)

        def print_val(val, totlen):

            if val is None:
                return center_str("None", totlen)
            else:
                if hasattr(val, "__len__"):
                    if type(val) == str:
                        if len(val) > totlen:
                            return center_str(val[:totlen-3] + "...", totlen)
                    if len(val) == 2:
                        if type(val[0]) == str:
                            lstr = val[0]
                        else:
                            lstr = f"{val[0]:.3f}"
                        if type(val[1]) == str:
                            rstr = val[1]
                        else:
                            rstr = f"{val[1]:.3f}"
                        return center_str(f"[{lstr}, {rstr}]", totlen)
                    else:
                        return center_str(str(val), totlen)
                else:
                    if type(val) == float:
                        return center_str(f"{val:.4f}", totlen)
                    elif type(val) == int:
                        return center_str(str(val), totlen)
                    elif type(val) == str:
                        if len(val) > totlen:
                            return center_str(val[:totlen-4] + "...", totlen)
                        return center_str(val, totlen)
                    elif type(val) == bool:
                        if val:
                            return center_str("TRUE", totlen)
                        else:
                            return center_str("FALSE", totlen)
                    else:
                        return center_str(str(val), totlen)

        def _print_fitted_params(pstr, pars):
            for i, key in enumerate(pars.keys()):
                if type(pars[key]) == ilex.pyfit._posterior:
                    pstr += center_str(key, strpad) + center_str(f"{pars[key].val:.4f}", strpad) + center_str(f"+{pars[key].p:.4f}/-{pars[key].m:.4f}", strpad) + "\n"
                else:
                    pstr += center_str(key, strpad) + center_str(pars[key], strpad)

            return pstr
                

        
        #create string outlining parameters

        outstr = "\n-> FRB parameters:\n\n"
        outstr += strcol(center_str("## Pars ##", strpad), 'cyan') + center_str("default", strpad) + center_str("latest", strpad) + "\n"
        outstr += "-"*(3*strpad) + "\n"
        for key in _G.p.keys():
            defval = getattr(self.par, key)
            instval = getattr(self.this_par, key)
            outstr += center_str(key, strpad) + print_val(defval, strpad) + print_val(instval, strpad) + "\n"
        
        outstr += "\n"
        outstr += strcol(center_str("## MetaPars ##", strpad), 'magenta') + center_str("default", strpad) + center_str("latest", strpad) + "\n"
        outstr += "-"*(3*strpad) + "\n"
        for key in _G.mp.keys():
            defval = getattr(self.metapar, key)
            instval = getattr(self.this_metapar, key)
            outstr += center_str(key, strpad) + print_val(defval, strpad) + print_val(instval, strpad) + "\n"

        if not self.verbose:
            return outstr

        outstr += "\n"
        outstr += strcol(center_str("## HyperPars ##", strpad), 'yellow') + center_str("values", strpad) + "\n"
        outstr += "-"*(2*strpad) + "\n"
        for key in _G.hp.keys():
            hval = getattr(self, key)
            outstr += center_str(key, strpad) + print_val(hval, strpad) + "\n"
        
        # print fitted params 
        if len(self.fitted_params) > 0:
            

            keys = self.fitted_params.keys()

            if "RM" in keys:
                outstr += "\n"
                outstr += strcol(center_str("## RM Pars ##", strpad), 'lgreen') + center_str("values", strpad) + center_str("errors", strpad) + "\n"
                outstr += "-"*(3*strpad) + "\n"
                outstr = _print_fitted_params(outstr, self.fitted_params['RM'])

            if "tscatt" in keys:
                outstr += "\n"
                outstr += strcol(center_str("## tscatt Pars ##", strpad), 'lred') + center_str("values", strpad) + center_str("errors", strpad) + "\n"
                outstr += "-"*(3*strpad) + "\n"
                outstr = _print_fitted_params(outstr, self.fitted_params['tscatt']) 

            if "scintband" in keys:
                outstr += "\n"
                outstr += strcol(center_str("## scintband Pars ##", strpad), 'lyellow') + center_str("values", strpad) + center_str("errors", strpad) + "\n"
                outstr += "-"*(3*strpad) + "\n"
                outstr = _print_fitted_params(outstr, self.fitted_params['scintband']) 



        return outstr





    def __deepcopy__(self, memo):
        """
        deepcopy instance of class

        [Does copy]
        Parameters, weights, filenames, fitted results and memory maps

        [Does not copy]
        data products such as _t, _f, _ds etc.

        """

        # create new instance
        frb = FRB()

        # add to memo to avoid recursion
        memo[id(self)] = frb

        # copy attributes
        for key, val in self.__dict__.items():
            if key not in _G.dc_exclude:
                setattr(frb, key, deepcopy(val, memo))
            else:
                setattr(frb, key, val)


        return frb











    def get_filepaths(self, stk = None):
        """
        Return filepath of stokes dynamic spectrum
        """

        if stk is None:
            return self._data_files
        else:
            return self._data_files[f"ds{stk}"]

    def get_hyperpars(self):
        """
        Returns list of hyperparams
        
        """
        hyperpar = {}

        for key in _G.hp:
            hyperpar[key] = getattr(self, key)
        
        return hyperpar



    




    ##===============================================##
    ##            Further FRB processing             ##
    ##===============================================##




    def burstinfo(self, **kwargs):
        """
        Measure properties of burst, including S/N, burst W, bandwidth and Fluence
        

        Returns
        -------
        prop : dict
            dictionary of burst properties \n
            [snr]: integrated S/N \n
            [pnsr]: peak S/N \n
            [w]: effective burst width (calculated from t_crop) \n
            [bw]: effectove bandwidth (calculated from f_crop) \n
            [fluence]: fluence of burst \n
            [centroid]: Centroid of burst
        """
 
        # init pars
        self._load_new_params(**kwargs)

        if not self._iserr():
            print("'terr_crop' must be specified to calculate S/N")
            return None
        
        data = self.get_data(['tI', 'dsI'], get = True, **kwargs)

        prop = {}

        # calculate SNR
        prop['snr'] = np.sum(data['tI']) / (data['tIerr'] * data['tI'].size**0.5)

        # calculate peak SNR
        prop['psnr'] = np.max(data['tI']) / data['tIerr']
        
        # width
        dt = data['time'][1] - data['time'][0]
        prop['w'] = data['time'][-1] - data['time'][0] + dt 

        prop['bw'] = self.this_par.bw
        prop['bw_eff'] = self.this_par.bw * float(np.where(~np.isnan(data['dsI'][:,0]))[0].size)/float(data['dsI'].shape[0])
        prop['cfreq'] = self.this_par.cfreq

        # fluence 
        # prop['fluence'] = self.this_metapar.tN * self.this_metapar.fN * np.nansum(data['dsI']) * prop['w']
        prop['fluence'] = np.sum(data['tI']) * prop['w']

        scal_fluence = np.sum(data['tI'])
        cent_cumsum = np.cumsum(data['tI']) - scal_fluence/2
        cent_samp = np.argmin(np.abs(cent_cumsum))
        prop['centroid'] = data['time'][cent_samp]

        prop['int_flux'] = scal_fluence

        units = {'snr': None, 'psnr': None, 'w': '[ms]', 'bw': '[MHz]', 'bw_eff': '[MHz] (w/o flagged chans)', 'cfreq': '[MHz]', 
                    'fluence': '[arbitrary]', 'centroid': '[ms]', 'int_flux': '[arbitrary]'}

        # print out info
        print("\n ---- Burst Properties ---- \n")
        for key in prop.keys():
            unit = units[key]
            if unit is None:
                unit = ""
            print(f"{key}".ljust(10) + ": " + f"{prop[key]:.4f}".ljust(15) + f"{unit}")
        print("\n")


        return prop










    ## [ FIND FRB PEAK AND TAKE REGION AROUND IT ] ##
    def find_frb(self, method = "fluence", mode = "min", sigma: int = 5, rms_guard: float = 10, rms_width: float = 50, 
                    rms_offset: float = 60, yfrac: float = 0.95, w: float = 30.0, stDev: int = 0,
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
        w: float
            initial width of data in [ms] centered at the peak that will be used to estimate 
            FRB bounds
        stDev: int
            HWFM [stDev * dt (ms)] of gaussian smoothing kernel to apply in time, if stDev = 0 will be skipped, by default 0
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

        # if 't_crop' not in kwargs.keys():
        kwargs['t_crop'] = ["min", "max"]
        kwargs['terr_crop'] = None
        f_crop = None
        
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
            
        # init pars
        self._load_new_params(**kwargs)

        # get data   
        self.get_data("tI", **kwargs)
        if not self._isdata():
            return None

        # make smaller buffer of data
        if w is not None:
            log(f"Searching w [{w}] ms around peak of burst", lpf_col = self.pcol)
            peak = np.argmax(self._t['I'])
            itcrop = [peak * self.this_par.dt - w/2, peak * self.this_par.dt + w/2]

            # guard to catch boundaries
            w = int(w / self.this_par.dt)
            tIstart = peak - w//2
            if tIstart < 0:
                tIstart = 0
            tIend = peak + w//2
            if tIend == self._t['I'].size:
                tIend = self._t['I'].size - 1

            tI = self._t['I'][tIstart:tIend]
            w_ref = tIstart
        else:
            log(f"Searching full time series", lpf_col = self.pcol)
            tI = self._t['I']
            w_ref = 0

        

        # now choose method of finding burst bounds
        log(f"Searching for Width using: {method} method", lpf_col=self.pcol)
        if method == "sigma":
            ms2phase = lambda x : x / (tI.size * self.this_par.dt)
            ref_ind, lw, rw = find_optimal_sigma_width(tI = tI, sigma = sigma,
                                rms_guard = ms2phase(rms_guard), 
                                rms_width = ms2phase(rms_width),
                                rms_offset = ms2phase(rms_offset))
            
            if ref_ind is None:
                return (None,) * 4
            
            log("Setting zero point reference [t_ref] to PEAK of burst", lpf_col = self.pcol)
        
        elif method == "fluence":

            # create copy of instance
            temp_frb = deepcopy(self)
            temp_frb.set(**kwargs)

            # ref_ind, lw, rw = find_optimal_fluence_width(tI = tI, yfrac = yfrac, mode = mode)
            _, f_crop, ref_ind, lw, rw = findfrb_fluence(temp_frb, yfrac = yfrac, stDev = stDev,
                                            itcrop = itcrop, ifcrop = self.this_par.f_lim, mode = mode, _iter = 0)

            # add buffer to f_crop 
            fbuff = (1.0-yfrac) * (f_crop[1] - f_crop[0])
            f_crop[0] -= fbuff
            f_crop[1] += fbuff
            if f_crop[0] < temp_frb.par.f_lim[0]:
                f_crop[0] = temp_frb.par.f_lim[0]
            if f_crop[1] > temp_frb.par.f_lim[1]:
                f_crop[1] = temp_frb.par.f_lim[1]

            # iterate with refine initial crops
            _, f_crop, ref_ind, lw, rw = findfrb_fluence(temp_frb, yfrac = yfrac, stDev = stDev,
                                            itcrop = itcrop, ifcrop = f_crop, mode = mode, _iter = 1)

            log("Setting zero point reference [t_ref] to EFFECTIVE CENTROID of burst", lpf_col = self.pcol)
        
        else:
            log(f"Undefined method [{method}].. Aborting!", lpf_col = self.pcol, stype = "err")
            return (None,)*4


        # Calculate new t_crop and t_ref relative to full time series dataset
        t_ref = (w_ref + ref_ind) * self.this_par.dt
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
        # self.metapar.set_metapar(terr_crop = [-rms_offset - rms_width, -rms_offset])
        if dt_from_peak_sigma is not None:
            self.metapar.set_metapar(tN = kwargs['tN'])
        self.par.set_par(t_ref = t_ref)
        if f_crop is not None:
            self.metapar.set_metapar(f_crop = f_crop)

        print("New t_crop: [{:.4f}, {:.4f}]".format(t_crop[0],t_crop[1]))
        print("New f_crop: [{:.4f}, {:.4f}]".format(*f_crop))
        # print(f"Setting terr_crop: [{-rms_offset - rms_width:.4f}, {-rms_offset:.4f}]")
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

        return t_crop, t_ref, tN, f_crop
    









    ##===============================================##
    ##                Plotting Methods               ##
    ##===============================================##
    def plot_data_on_axes(self, data = "dsI", ax = None, stk_debias = False, stk_ratio = False, stk_sigma = None,
                     filename: str = None, **kwargs):
        """
        Plot data onto a given axes, if ax is not specified, will make a seperate figure.

        Parameters
        ----------
        data : str, optional
            type of data to plot, by default "dsI"
        ax : axes, optional
            Axes object to plot data into
        stk_debias : bool, optional
            If True, Any L or P data plotted will be debiased
        stk_ratio : bool, optional
            If True, any t or f data will be converted to X/I and plotted
        stk_sigma, optional
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
        pdat = self.get_data(data_list = data, get = True, stk_debias = stk_debias, 
                                stk_ratio = stk_ratio, stk_sigma = stk_sigma,
                                **kwargs)

        if not self._isdata():
            return None

        # plot 
        fig = plot_data(pdat, data, ax = ax, plot_type = self.plot_type, showzaps = self.show_dynzaps)

        if self.save_plots:
            if filename is None:
                filename = f"{self.par.name}_{data}.png"
            plt.savefig(filename)

        if self.show_plots:
            plt.show()

        self._save_new_params()

        return fig






    

    def plot_data(self, data = ['tI', 'dsI'], layout = "vertical", stk_debias = False, stk_ratio = False, 
                        stk_sigma = None, plot_weights = False, plot_labels = True, figsize = None, filename = None, **kwargs):
        """
        Master function for ploting basic Stokes I, Q, U, V, L and P products.
        The ''fig'', ''ax'' instances can be returned in-case further alterations to the plot are nessessary.
        
        Parameters
        ----------
        data: list[str]
            list of products to plot, this includes \n
            ['tI', 'tQ', 'tU', 'tV', 'tL', 'tP'] for the Stokes time-series products \n
            ['fI', 'fQ', 'fU', 'fV', 'fL', 'fP'] for the Stokes spectrum products \n
            ['dsI', 'dsQ', 'dsU', 'dsV'] for the Stokes dynamic spectrum products \n\n
            Additionally, you can specify combinations of products \n
            ['dsI', 'tI'] will plot both the Stokes I dynamic spectrum and time-series \n
            ['tIQ'] will plot both the Stokes I and Q time-series \n\n
            You can specify sets of products as well \n
            ['ds'] will plot all Stokes dynamic spectra ['IQUV'], ['t', 'f'] will plot all the Stokes time-series/spectra ['IQUVLP']\n
            ['I'] will plot all Stokes I products ['dsI', 'tI', 'fI']\n\n
            Any combination of the above products can be used to make whatever plot you like. Finally\n
            ['all'] will plot all Stokes products.
        layout: str
            layout can be either ['vertical', 'horizontal'] \n
            'vertical' will plot all avaliable dynamic spectrum stacked vertically, all time-series products will be plotted above this stack\n
            in the same axes. Each spectrum product will be plotted seperately in an axes adjacent to the corrosponding stokes dynamic spectrum.\n
            'horizontal' will plot all avaliable dynamic spectrum in a row, all spectrum products will be plotted on the far-right beside the dynspec row\n
            in the same axes. Each time-series product will be plotted seperately in an axes above the corrosponding stokes dynamic spectrum.\n\n
            In the case where only Stokes time-series/spectrum are plotted (no 'ds'), all time-series/spectrum products are plotted on the same axes.\n
            The 'layout' argument will not affect the resulting plot.
        stk_debias: bool
            debias Stokes L and P products, by default False
        stk_ratio: bool
            plot Stokes fractions, by default False
        stk_sigma: float
            If stk_ratio = True, any time/freq sample that does not meet the stk_ratio threshold will be masked, by default None
        plot_weights: bool
            If true, The time/freq weights will be normalized and plotted with their respective Stokes products, by default False
        plot_labels: bool
            add labels/legends to the axes to denote each product, by default True
        filename: str
            save the figure as a .png file, by default None
        **kwargs:
            ILEX parameters
        
        Returns
        -------
        fig: matplotlib.pyplot.Figure
            Figure instance of plot
        ax: dict[matplotlib.pyplot.Axes]
            Dictionary of key:axes, key is the stokes product plotted on the specific axes, 't0', 'f0'\n
            denote axes where multiple time/freq products have been plotted
        """

        def remove_item_from_list(_list, item):
            if item in _list:
                _list.pop(_list.index(item))
            return

        def get_ax_from_layout(figlayout, _type):

            return [figcol for figrow in figlayout for figcol in figrow if _type in figcol]

        def figlayout_replace(figlayout, item, item_replace):

            for i in range(len(figlayout)):
                for j in range(len(figlayout[0])):
                    if figlayout[i][j] == item:
                        figlayout[i][j] = item_replace


        valid_data = ['tI', 'tQ', 'tU', 'tL', 'tV', 'fI', 'fQ', 'fU', 'fL', 'fV',
                      'dsI', 'dsQ', 'dsU', 'dsV', 'fP', 'tP']

        ds_sizes = [6, 4, 3, 2.5]
        tf_sizes = [2.5, 2.0, 1.5, 1.25]
        legend_loc = "upper right"
        # figure_scale = 1.5

        if layout == "vertical":
            figlayout = [['t0', '.'], ['dsI', 'fI'], ['dsQ', 'fQ'], ['dsU', 'fU'], ['dsV', 'fV']]
        elif layout == "horizontal":
            figlayout = [['tI', 'tQ', 'tU', 'tV', '.'], ['dsI', 'dsQ', 'dsU', 'dsV', 'f0']]
        else:
            print(f"layout = {layout} is not a valid layout!, must be [vertical, horizontal]!")
            sys.exit()

        # check if dat is str, if so convert to list 
        if type(data) == str:
            data = [data]

        # process data product list
        full_data = []
        for dat in data:
            if dat in ['I', 'Q', 'U', 'V']:
                full_data += [f"t{dat}", f"f{dat}", f"ds{dat}"]
                continue
            if dat in ['t', 'f']:
                full_data += [f"{dat}I", f"{dat}L", f"{dat}Q", f"{dat}U", f"{dat}V", f"{dat}P"]
                continue
            if dat == "ds":
                full_data += ["dsI", "dsQ", "dsU", "dsV"]
                continue
            if dat == "all":
                full_data += valid_data
                continue
            if (dat[0] in ['t', 'f']) and (len(dat) > 2):
                for s in dat[1:]:
                    if s in "ILQUVP":
                        full_data += [f"{dat[0]}{s}"]
                continue
            if (dat[0:2] == "ds") and (len(dat) > 3):
                for s in dat[2:]:
                    if s in "IQUV":
                        full_data += [f"{dat[0:2]}{s}"]
                continue
            if dat not in valid_data:
                print(f"[{dat}] is not a valid data product to plot!")
                return
            full_data += [dat]
        full_data = list(set(full_data))

        flags = {'t': False, 'f': False, 'ds': 0}   
        flags['ds'] = len([dat for dat in full_data if "ds" in dat]) 

        # alter figlayout according to data products the user wants to plot
        if layout == "vertical":
            idpop = 1
            for s in "IQUV":
                if not f"ds{s}" in full_data:
                    # remove row in figure
                    figlayout.pop(idpop)    # always in order

                    # remove data products
                    remove_item_from_list(full_data, f"ds{s}")
                    if flags['ds'] > 1:
                        remove_item_from_list(full_data, f"f{s}")
                else:
                    idpop += 1

            # remove L and P products
            if flags['ds'] > 1:
                for dat in ['fL', 'fP']:
                    remove_item_from_list(full_data, dat)    
                

        elif layout == "horizontal":
            idpop = 0
            for s in "IQUV":
                if not f"ds{s}" in full_data:
                    # remove column in figure
                    for i in range(len(figlayout)):
                        figlayout[i].pop(idpop) # again, always in order

                    # remove data products
                    remove_item_from_list(full_data, f"ds{s}")
                    if flags['ds'] > 1:
                        remove_item_from_list(full_data, f"t{s}")
                else:
                    idpop += 1
            
            # remove L and P products
            if flags['ds'] > 1:
                for dat in ['tL', 'tP']:
                    remove_item_from_list(full_data, dat)    
        
        # normalising 
        if stk_ratio:
            if layout == "vertical":
                remove_item_from_list(full_data, 'tI')
            else:
                remove_item_from_list(full_data, 'fI')


        # check if t or f items in full_data, also split time and freq products 
        # and count number of dynspec
        time_data = []
        freq_data = []
        for dat in full_data:
            if dat[0] == "t":
                flags['t'] = True
                time_data += [dat]
            if dat[0] == "f":
                flags['f'] = True
                freq_data += [dat]
            if dat[0:2] == "ds":
                time_data += [dat]

        # edge case, if no dynamic spectra are avaliable
        if flags['ds'] == 0:
            figlayout = []
            if flags['t']:
                figlayout += [['t0']]
            if flags['f']:
                figlayout += [['f0']]


        # further update figlayout incase t and or f products are absent
        if flags['ds'] > 0:
            if not flags['t']:
                figlayout.pop(0)
            if not flags['f']:
                for i in range(len(figlayout)):
                    figlayout[i].pop(-1)

        # check if only 1 t/f axes remains, if so rename
        tax = get_ax_from_layout(figlayout, "t")
        if len(tax) == 1:
            figlayout_replace(figlayout, tax[0], "t0")
            tax[0] = "t0"

        fax = get_ax_from_layout(figlayout, "f")
        if len(fax) == 1:
            figlayout_replace(figlayout, fax[0], "f0")
            fax[0] = "f0"

    
        # Get data
        if 't_crop' not in kwargs.keys():
            kwargs['t_crop'] = self.metapar.t_crop.copy()
        tcrop = kwargs['t_crop'].copy()
        kwargs['t_crop'] = self.get_ptcrop(tcrop)

        stk = self.get_data(time_data, stk_debias = stk_debias, 
                            stk_ratio = stk_ratio, stk_sigma = stk_sigma, get = True, **kwargs)
        kwargs['t_crop'] = tcrop.copy()
        tcrop = self.this_par.t_lim.copy()
        fstk = self.get_data(freq_data, stk_debias = stk_debias, stk_ratio = stk_ratio, 
                             stk_sigma = stk_sigma, get = True, **kwargs)
        fcrop = self.this_par.f_lim.copy()
        otcrop = self.this_par.t_lim.copy() 

        # combine data
        for key in fstk.keys():
            if key not in ['time']:
                if fstk[key] is None:
                    stk[key] = None
                else:
                    stk[key] = fstk[key].copy()
        del fstk

        update_figsize = False
        if figsize is None:
            update_figsize = True



        # update additional figure paramters including figsize according to data products
        if flags['ds'] > 0:
            n = flags['ds'] - 1
            # assume vertical first
            if update_figsize:
                figsize = [ds_sizes[n], (n + 1)*ds_sizes[n]]
            width_ratios = [ds_sizes[n]]
            height_ratios = [ds_sizes[n]] * (n + 1)
            
            if layout == "horizontal":
                if update_figsize:
                    figsize = figsize[::-1]
                width_ratios, height_ratios = height_ratios.copy(), width_ratios.copy()            
                
            if flags['t']:
                height_ratios += [tf_sizes[n]]
                if update_figsize:
                    figsize[1] += tf_sizes[n]
            if flags['f']:
                width_ratios += [tf_sizes[n]]
                if update_figsize:
                    figsize[0] += tf_sizes[n]
        
        else:
            tfnum = int(flags['t'] + flags['f'])
            if update_figsize:
                figsize = [10, tfnum * 5]
            width_ratios = [1]
            height_ratios = [1] * tfnum



        # create figure
        fig, ax = plt.subplot_mosaic(figlayout, figsize = figsize, 
                        gridspec_kw = {'width_ratios':width_ratios, 'height_ratios':height_ratios[::-1]})

        # axes sharing
        axdims = [len(figlayout), len(figlayout[0])]
        if flags['ds'] > 0:
            if flags['ds'] > 1:
                dsax = get_ax_from_layout(figlayout, "ds")
                for d in dsax[1:]:
                        ax[d].sharex(ax[dsax[0]])
                        ax[d].sharey(ax[dsax[0]])
            if flags['t']:
                if layout == "vertical":
                    ax['t0'].sharex(ax[figlayout[-1][0]])
                else:
                    tax = get_ax_from_layout(figlayout, "t")
                    ax[tax[0]].sharex(ax[figlayout[-1][0]])
                    for d in tax[1:]:
                        ax[d].sharex(ax[tax[0]])
                        ax[d].sharey(ax[tax[0]])
            if flags['f']:
                if layout == "horizontal":
                    ax['f0'].sharey(ax[figlayout[-1][0]])
                else:
                    fax = get_ax_from_layout(figlayout, "f")
                    ax[fax[0]].sharey(ax[figlayout[-1][0]])
                    for d in fax[1:]:
                        ax[d].sharey(ax[fax[0]])
                        ax[d].sharex(ax[fax[0]])
        

        # axes labeling
        for i in range(axdims[0]):
            for j in range(axdims[1]):
                # label
                figax = figlayout[i][j]
                if figax == ".":
                    continue
                if "t" in figax:
                    ax[figax].set_xlabel("Time [ms]")
                    ax[figax].set_ylabel("Flux [a.u]")
                
                if "f" in figax:
                    fxlab = "Freq [MHz]"
                    fylab = "Flux [a.u]"
                    if flags['ds'] > 0:
                        fxlab, fylab = fylab, fxlab
                    ax[figax].set_xlabel(fxlab)
                    ax[figax].set_ylabel(fylab)
                
                if "ds" in figax:
                    ax[figax].set_xlabel("Time [ms]")
                    ax[figax].set_ylabel("Freq [MHz]")

                # label x axis
                if (i != axdims[0]-1) and (flags['ds'] > 0):
                    ax[figax].get_xaxis().set_visible(False)
                
                if j != 0:
                    ax[figax].get_yaxis().set_visible(False)

        # plot data
        for dat in full_data:
            activate_legend = False # in case seperate axes are used for each product, make sure legend is made
            
            # plot time data
            if "t" in dat:
                if (layout == "horizontal") and (flags['ds'] > 1):
                    activate_legend = True
                    axi = ax[dat]
                else:
                    axi = ax['t0']
                plot(x = stk['time'], y = stk[dat], yerr = stk[f"{dat}err"],
                    ax = axi, plot_type = self.plot_type, color = _G.stk_colors[dat[-1]], 
                    label = dat[-1])

                # plot used time weights if applicable
                if plot_weights and (dat == "tI"):
                    xw = np.linspace(stk['time'][0], stk['time'][-1], 10000)
                    tW = self.par.tW.get_weights(x = xw)
                    if tW is not None:
                        tW = tW / np.max(tW) * np.nanmax(stk['tI'])
                        axi.plot(xw, tW, color = 'peru', linestyle = "--", linewidth = 1.0,
                                label = "tI weights")

            # plot freq data
            if "f" in dat:
                if (layout == "vertical") and (flags['ds'] > 1):
                    activate_legend = True
                    axi = ax[dat]
                    axfhandle = ax[dat]     # This handle is used later to get ylim of avaliable f axes
                else:
                    axi = ax['f0']
                    axfhandle = ax['f0']
                fx, fxerr = stk['freq'], None
                fy, fyerr = stk[dat], stk[f"{dat}err"]

                # freq weights
                weight_flag = False
                if plot_weights and (dat == "fI"):
                    fW = self.par.fW.get_weights(x = stk['freq'])
                    if fW is not None:
                        weight_flag = True
                        fwx, fwy = fx, stk[dat]
 
                if flags['ds'] > 0:
                    fx, fy = fy, fx
                    fxerr, fyerr = fyerr, fxerr
                    if weight_flag:
                        fwx, fwy = fwy, fwx
                plot(x = fx, y = fy, xerr = fxerr, yerr = fyerr, ax = axi,
                    plot_type = self.plot_type, color = _G.stk_colors[dat[-1]], label = dat[-1])
                if weight_flag:
                    axi.plot(fwx, fwy, color = 'peru', linestyle = "--", linewidth = 1.0,
                            label = "fI weights")

            if activate_legend and plot_labels:
                axi.legend(loc = legend_loc)

            # plot dynamic spectra
            if "ds" in dat:
                plot_dynspec(stk[dat], ax = ax[dat], extent = [*tcrop, *fcrop],
                    aspect = 'auto', interpolation = "none", showzaps = self.show_dynzaps)

                if plot_labels:
                    ax[dat].plot([], [], color = _G.stk_colors[dat[-1]], label = dat[-1])
                    ax[dat].legend(loc = legend_loc)


        # add last legends to combined t/f axes
        if plot_labels:
            if len(tax) == 1:
                ax['t0'].legend(*sort_legend(ax['t0']), loc = legend_loc)
            if len(fax) == 1:
                ax['f0'].legend(*sort_legend(ax['f0']), loc = legend_loc)


        # plot bounds of t_crop
        if otcrop != tcrop:
            if flags['t']:
                ylim = None
                for key in ax.keys():
                    if key[0] == 't':
                        if ylim is None:
                            ylim = ax[key].get_ylim()
                        ax[key].fill_between(otcrop, *ylim, color = "coral", 
                                            zorder = 0, alpha = 0.15)
                        ax[key].set_ylim(ylim)

        
        # fix xlims of spectra axes
        fmaxs, fmins = [], []
        for dat in full_data:
            if "f" in dat:
                fmaxs += [np.nanmax(stk[dat])]
                fmins += [np.nanmin(stk[dat])]
        if (len(fmaxs) > 0) and (len(fmins) > 0):
            w = max(fmaxs) - min(fmins)
            lims = [min(fmins) - 0.1*w, max(fmaxs) + 0.1*w]
            if flags['ds'] > 0:
                axfhandle.set_xlim(lims)
            else:
                axfhandle.set_ylim(lims)

        
        # remove empty axes
        for key in ax.keys():
            if flags['ds'] > 0:
                if ("t" in key) and (key[-1] in "IQUV"):
                    if (layout == "horizontal") and (f"ds{key[-1]}" in full_data):
                        if key not in full_data:
                            ax[key].clear()
                            ax[key].set_axis_off()
                    continue 
                if ("f" in key) and (key[-1] in "IQUV"):
                    if (layout == "vertical") and (f"ds{key[-1]}" in full_data):
                        if key not in full_data:
                            ax[key].clear()
                            ax[key].set_axis_off()
                    continue

        # Render plot
        fig.tight_layout()
        if flags['ds'] > 0:
            fig.subplots_adjust(hspace = 0, wspace = 0)
        

        if self.save_plots:
            if filename is None:
                file_suffix = ""
                label_comps = {'f':"", 't':"", "ds":""}

                # construst filename
                for d in valid_data:
                    if d in full_data:
                        label_comps[d[:-1]] += d[-1]
                
                for prt in ['f','t','ds']:
                    if len(label_comps[prt]) > 0:
                        file_suffix += prt + label_comps[prt] + "_"

                filename = f"{self.par.name}_{file_suffix[:-1]}.png"
            fig.savefig(filename)

        if self.show_plots:
            plt.show()

        self._save_new_params()

        return fig, ax



    def get_PA(self, Ldebias = 0.0, rad2deg = False, **kwargs):
        """
        Get PAs for FRB crop in time

        Parameters
        ----------
        Ldebias : float
            if non-zero, will debias the PA results
        rad2deg : bool
            if True, convert PA units from "rad" to "deg", by default True
        kwargs : dict
            FRB parameters
        
        Returns
        -------
        PA : np.ndarray 
            Polarization position angle 1D numpy array
        PAerr : np.ndarray
            Polarisation position angle 1D numpy array errors
        t : np.ndarray
            time values
        """

        # init pars
        self._load_new_params(**kwargs)
    

        ##====================##
        ##     do fitting     ##
        ##====================##

        # get data 
        data = self.get_data(['tI', 'tQ', 'tU'], get = True, **kwargs)

        Ldebias_flag = False
        if (Ldebias > 0.0) and self._iserr():
            Ldebias_flag = True
        
        # Get PAs
        if Ldebias_flag:
            PA, PAerr = calc_PAdebiased(data, Ldebias_threshold = Ldebias, 
                                        rad2deg = rad2deg)
        
        else:
            PA, PAerr = calc_PA(Q = data['tQ'], U = data['tU'], 
                                Qerr = data['tQerr'], Uerr = data['tUerr'], 
                                rad2deg = rad2deg)
        
        return PA, PAerr, data['time']

        












    def zap_channels(self, chans: str = None, zapzeros: bool = False, zapzerosmargin: float = 1e-5, 
                    zapsigma: float = None, stDev: int = 0, auto: bool = False, auto_tN: int = 1000,
                    auto_iter: int = 1, resetzap: bool = False, interactive = False, overwrite: bool = True, **kwargs):
        """
        Zap channels (uses Stokes I freq spectrum)

        Parameters
        ----------
        chans : str, optional
            Channels to zap, by default None
        zapzeros : bool, optional
            zap channels at or close to zero, by default False
        zapzerosmargin : float
            margin close to zero at which to zap channels, ratio of max channel flux
        zapsigma : float
            zap channels above a SNR threshold
        stDev : int
            Standard deviation in integer samples for gaussian smoothing, used to smooth data
            when using zapsigma as a SNR threshold
        resetzap : bool, optional
            reset channels zapped, by default False
        interactive : bool, optional
            Enable interactive zapping mode
        overwrite : bool, optional
            Overwrite string of zapped channels with new set of zapped channels, by default True
        auto : bool, optional
            Toggle automatic channel zapping using a statistical median approach [see Dial et al, 2026a]. zapsigma is used as the threshold parameter
            if not specified a value of zapsigma = 3.0 will be used by default.
        auto_tN : int
            Downsampling performed on data in time before statistical automatic channel zapping
        auto_iter : int
            Number of iterations to perform statistical median channel zapping
        **kwargs : dict
            Standard frb parameters that can be overidden (zapchan cannot be overidden)
        """

        if "zapchan" in kwargs.keys():
            del kwargs['zapchan']
        
        # init pars
        self._load_new_params(**kwargs)

        zapchan = self.metapar.zapchan
        if zapchan is None:
            zapchan = ""

        if resetzap:
            fcrop = self.metapar.f_crop
            if 'f_crop' in kwargs.keys():
                fcrop = kwargs['f_crop'].copy()
            kwargs['f_crop'] = ['min', 'max']
            data = self.get_data('fI', get = True, zapchan = "", **kwargs)
            zapchan = get_zapstr(data['fI'], self.par.get_freqs())
            kwargs['f_crop'] = fcrop.copy()
        
        if chans is not None:
            zapchan += ("," + chans)
        
        if zapzeros:
            data = self.get_data('fI', get = True, zapchan = zapchan, **kwargs)
            data['fI'][np.isnan(data['fI'])] = np.max(data['fI'][~np.isnan(data['fI'])])
            data['fI'][np.abs(data['fI']/np.max(np.abs(data['fI']))) < zapzerosmargin] = np.nan
            zapchan += ("," + get_zapstr(data['fI'], self.par.get_freqs()))

        if zapsigma is not None:
            # zap to a SNR threshold, this requires errors 
            if not self._iserr():
                ValueError("Must specify [terr_crop] before applying a SNR threshold!")
            data = self.get_data('fI', get = True, zapchan = zapchan, **kwargs)
            fIsmooth = gaussian_smooth(data['fI'], stDev)
            mask = fIsmooth < data['fIerr'] * zapsigma 
            nanchans = np.ones(data['fI'].size)
            nanchans[mask] = np.nan 
            zapchan += ("," + get_zapstr(nanchans, data['freq'])) 

            # make plot
            if self.save_plots or self.show_plots:
                fig_smth, ax_smth = plt.subplots(1, 1, figsize = (10, 7))
                plot(x = data['freq'], y = data['fI'], yerr = data['fIerr'], ax = ax_smth, plot_type = "scatter",
                     color = 'k', label = "Stokes I")
                fIsmooth[mask] = np.nan
                ax_smth.plot(data['freq'], fIsmooth, color = 'r', label = "Smooted I")
                ax_smth.set_xlabel("Freq [MHz]", fontsize = 16)
                ax_smth.set_ylabel("Flux Density (arb.)", fontsize = 16)
                ax_smth.set_title(f"Channel zapping with threshold: {zapsigma} and stDev: {stDev}")

                if self.save_plots:
                    fname = self.this_par.name + "_zapsigma.png"
                    log(f"Saving plot of zapping using gaussian smoothing on spectra saved as: {fname}")
                    plt.savefig(fname)

                if self.show_plots:
                    plt.show()

        if auto: 
            # zap using a statistical approach
            data = self.get_data("dsI", zapchan = zapchan, **dict_edit_and_copy(kwargs, {'tN':auto_tN}), 
                                get = True)
            if zapsigma is None:
                zapsigma = 3.0
            flagchan = medrms_chanflag(data['dsI'], threshold = zapsigma, tN = 1, iter = auto_iter)
            nanchans = np.ones(data['dsI'].shape[0])
            nanchans[flagchan] = np.nan
            zapchan += ("," + get_zapstr(nanchans, data['freq']))

        # make sure there are no commas in illegal places
        test_zapchan = zapchan.strip()
        if len(test_zapchan) > 0:
            if test_zapchan[0] == ",":
                test_zapchan = test_zapchan[1:]
        if len(test_zapchan) > 0:
            if test_zapchan[-1] == ",":
                test_zapchan = test_zapchan[:-1]
            
            
        if interactive:
            data = self.get_data('dsI', zapchan = test_zapchan, **kwargs, get = True)
            data['dsI'][np.isnan(data['dsI'][:,0])] = 0.0
            test_zapchan = ZapInteractive(data['dsI'], data['freq'], data['time'], zapchan = test_zapchan)
        
        # save
        if overwrite:
            self.metapar.zapchan = test_zapchan

        return test_zapchan








    def plot_stokes(self, ax = None, stk_debias = False, stk_sigma = 2.0, 
            stk_type = "f", stk2plot = "IQUV", stk_ratio = False, filename: str = None, **kwargs):
        """
        Plot Stokes data, by default stokes I, Q, U and V data is plotted

        Parameters
        ----------
        ax: _axes_
            matplotlib.pyplot.axes object to plot to, default is None
        stk_debias : bool, optional
            Plot stokes L and/or P debias, by default False
        stk_sigma : float, optional
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
            stk_debias = False

        # data container for plotting
        pstk = {}

        if not stk_type in "ft":
            log("stk_type can only be t or f", lpf_col = self.pcol, stype = "err")

        # plot
        fig = plot_stokes(data, Ldebias = stk_debias, stk_type = stk_type,
                    sigma = stk_sigma, stk2plot = stk2plot, stk_ratio = stk_ratio,
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

        # check of off-pulse region has been given
        err_flag = True
        if self.this_metapar.terr_crop is None:    # this essentially ignores the off-pulse region when plotting
            err_flag = False
            terrcrop = tcrop.copy()

        # get crop in time and frequency, these will be used to draw the bounds of the crops in a larger
        # dynamic spectrum
        tcrop = self.this_metapar.t_crop.copy()
        fcrop = self.this_metapar.f_crop.copy()
        if err_flag:
            terrcrop = self.this_metapar.terr_crop.copy()

        # combine crops, these will be crops of the full dynamic spectrum  with bound markers
        fcrop_ds = [0.0, 1.0]       # by default take full bandwidth
        tpad = 50 / (self.par.t_lim[-1] - self.par.t_lim[0])   # by default we will pad time by 100ms

        

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
                                                             *self.this_par.f_lim],
                                                             showzaps = self.show_dynzaps)
        plt.xlabel("Time [ms]")
        plt.ylabel("Freq [MHz]")

        tcrop, fcrop = self.par.phase2lim(t_crop = tcrop, f_crop = fcrop)

        if err_flag:
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

        # plot peak of crop
        peak = np.argmax(np.nanmean(self._ds[stk], axis = 0))
        plt.plot([self._time[peak]]*2, self.this_par.f_lim, color = "springgreen", linestyle = "--", label = "peak")


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
    def fit_scintband(self, method = "bayesian", priors: dict = None, statics: dict = None, 
                     fit_params: dict = None, redo = False, filename: str = None, n: int = None, 
                     intrinsic_removal: str = None, maxlag: float = None, **kwargs):
        """
        Fit for, Find and plot Scintillation bandwidth in FRB. Optionally, normalize/subtract the intrinsic fitted burst structure
        to retrieve only the scintillation features.

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
        intrinsic_removal : str, optional
            Method for removing intrinsic broad-band burst structures from spectra, by default None \n
            [None, 'none'] -> No removal \n
            ['subtract'] -> Subtract off the fitted broadband burst features from the spectra \n
            ['normalize'] -> Normalize spectra by the broaband fitted burst (divide) 
        maxlag : float, optional
            Maximum frequency [MHz] lag to fit for scintillation bandwidth, by default None i.e. full freq lag range
        n : float, optional
            Polynomial order, by default None, if not given then the intrinsic broad-band features will not be removed from the spectra

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

        if intrinsic_removal is None:
            intrinsic_removal = 'none'
        if intrinsic_removal not in ['none', 'subtract', 'normalize']:
            log(f"{intrinsic_removal} must be one of the following: ['none', 'subtract', 'normalize']", stype = 'err')
            return None

        def remove_intrinsic(I, method, n):

            if method == 'none':
                return I, None
            if method == 'subtract':
                return residuals(I, n = n)
            if method == 'normalize':
                return mean_normalize(I, n = n)


            
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
            y, yfit = remove_intrinsic(self._f['I'], 
                            method = intrinsic_removal, n = n)
            yrms = sumfunc(y**2)
        else:
            y = self._f['I']
            yrms = None

        # set nans to zero
        
        y = acf(y)

        # in case zapping is involved
        mask = np.isnan(y)

        # lags
        x = np.linspace(self.this_par.df, self.this_par.bw - self.this_par.df,
                         y.size)

        # maxlag
        if maxlag is not None:
            mask[np.abs(x) > maxlag] = True

        # create instance of fitting
        yerr = None
        p = fit(x = x[~mask], y = y[~mask], yerr = None, func = lorentz_yshifted, prior = priors,
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
        print(p)



        ##===================##
        ##   do plotting     ##
        ##===================##  
        if self.save_plots or self.show_plots:    

            if n is not None: 
                plt.figure(figsize = (8, 6))
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

            if maxlag is not None:
                if maxlag < np.max(x):
                    for a in fig.get_axes():
                        a.set_xlim([0, maxlag])

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



## [ PLOT LORENTZ OF CROP ] ##
    def fit_scintband2(self, method = "bayesian", priors: dict = None, statics: dict = None, 
                     fit_params: dict = None, redo = False, filename: str = None, n: int = None, 
                     intrinsic_removal: str = None, maxlag: float = None, **kwargs):
        """
        Fit for, Find and plot Scintillation bandwidth in FRB. Optionally, normalize/subtract the intrinsic fitted burst structure
        to retrieve only the scintillation features.

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
        intrinsic_removal : str, optional
            Method for removing intrinsic broad-band burst structures from spectra, by default None \n
            [None, 'none'] -> No removal \n
            ['subtract'] -> Subtract off the fitted broadband burst features from the spectra \n
            ['normalize'] -> Normalize spectra by the broaband fitted burst (divide) 
        maxlag : float, optional
            Maximum frequency [MHz] lag to fit for scintillation bandwidth, by default None i.e. full freq lag range
        n : float, optional
            Polynomial order, by default None, if not given then the intrinsic broad-band features will not be removed from the spectra

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

        if intrinsic_removal is None:
            intrinsic_removal = 'none'
        if intrinsic_removal not in ['none', 'subtract', 'normalize']:
            log(f"{intrinsic_removal} must be one of the following: ['none', 'subtract', 'normalize']", stype = 'err')
            return None

        def remove_intrinsic(I, method, n):

            if method == 'none':
                return I, None
            if method == 'subtract':
                return residuals(I, n = n)
            if method == 'normalize':
                return mean_normalize(I, n = n)


            
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
            y, yfit = remove_intrinsic(self._f['I'], 
                            method = intrinsic_removal, n = n)
            yrms = sumfunc(y**2)
        else:
            y = self._f['I']
            yrms = None

        # set nans to zero
        
        y = acf(y, outs = 'all')

        # in case zapping is involved
        mask = np.isnan(y)
        mask[mask.size//2] = True

        # lags
        x = np.linspace(-self.this_par.bw + self.this_par.df, self.this_par.bw - self.this_par.df,
                         y.size)

        # maxlag
        if maxlag is not None:
            mask[np.abs(x) > maxlag] = True

        # create instance of fitting
        yerr = None
        p = fit(x = x[~mask], y = y[~mask], yerr = None, func = lorentz_yshifted, prior = priors,
                static = statics, fit_keywords = fit_params, method = method,
                residuals = self.residuals, plotPosterior = self.plotPosterior)
        p.set_plot_vars(plot_data_kwargs = {'alpha':0.5})

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
        print(p)



        ##===================##
        ##   do plotting     ##
        ##===================##  
        if self.save_plots or self.show_plots:    

            if (n is not None) and (intrinsic_removal != 'none'): 
                plt.figure(figsize = (8, 6))
                plt.plot(self._freq, self._f['I'], 'k', label = "STOKES I spectra")
                plt.plot(self._freq, yfit(np.arange(self._f['I'].size)), 'r--', label = "STOKES I fit")
                plt.xlabel("Freq [MHz]")
                plt.ylabel("Flux (arb.)")
                plt.title(f"polyfit, n = {n}")
                plt.legend()

                if self.save_plots:
                    if filename is None:
                        filename_s = f"{self.par.name}_fit_scintband_broad_poly_model.png"
                    else:
                        filename_s = filename + "_broad_poly_model.png"
                    
                    plt.savefig(filename_s)
                

            fig = p.plot(xlabel = "Freq [MHz]", ylabel = "Norm acf", show = False)
            xnans, ynans = x.copy(), y.copy()
            xnans[mask] = np.nan
            ynans[mask] = np.nan
            axs = fig.get_axes()
            if self.residuals:
                axs[1].plot(xnans, ynans - p.get_model(xnans)[1], c = 'k', alpha = 0.5, zorder = 0)
                axs[1].plot(x, y - p.get_model(x)[1], c = [0.6,0.6,0.6], alpha = 0.7, zorder = 0)
            axs[0].plot(xnans, ynans, c = 'k', alpha = 0.5, zorder = 0)
            axs[0].plot(x, y, c = [0.6,0.6,0.6], alpha = 0.7, zorder = 0)

            if maxlag is not None:
                if maxlag < np.max(x):
                    for a in fig.get_axes():
                        a.set_xlim([-maxlag, maxlag])

            if self.save_plots:
                if filename is None:
                    filename_s= f"{self.par.name}_fit_scintband.png"
                else:
                    filename_s = filename + ".png"
                
                plt.savefig(filename_s)


            if self.show_plots:
                plt.show()


        # update instance par
        self._save_new_params()

        return p



    ## [ FIT SCATTERING TIMESCALE ] ##
    def fit_tscatt(self, method = "bayesian", npulse = 1, fitmode = "abs", priors: dict = None, statics: dict = None, 
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
        fitmode : str
            Mode for fitting Multiple gaussians \n
            [abs]: The position [mu] of each gaussian pulse is an absolute position \n
            [relative]: The position of the first gaussian pulse is an absolute position whilst all other gaussin positions are \n
                        relative to [mu1].
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


        ## fitmode ##
        if fitmode == "abs":
            tscattfunc = scatt_pulse_profile
        elif fitmode == "relative":
            tscattfunc = scatt_pulse_profile_relative
        else:
            ValueError(f"fitmode = {fitmode} is not supported!")

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
        # this will be used to pass the function and data to fit even if "least squares"
        # is being used and the likelihood obj itself is not used
        likelihood = None
        if method == "bayesian":
            likelihood = tscattLikelihood(x = x, y = y, yerr = err, npulse = npulse, 
                                            fitmode = fitmode)

        p = fit(x = x, y = y, yerr = err, likelihood = likelihood, func = tscattfunc,
                prior = priors, static = statics, fit_keywords = fit_params, method = method,
                residuals = self.residuals, plotPosterior = self.plotPosterior) 

        # make sure we don't sample 0 for select priors
        if method == "least squares":
            for key in p.keys:
                if key == "sigma":
                    continue
                if key not in priors.keys():
                    if ("sig" in key) or ("tau" in key) or (key[0] == "a"):
                        p.bounds[key] = [0.0001, math.inf]

        # fit 
        p.fit(redo = redo)

        # set to fitted params
        self.fitted_params['tscatt'] = p.get_posteriors()
        self.fitted_params['tscatt']['npulse'] = npulse
        self.fitted_params['tscatt']['fitmode'] = fitmode
        
        # print best fit parameters
        if self.verbose:
            p.stats()
        print(p)

        # extra printing
        print("Additional info:")
        print(f"cfreq: {self.this_par.cfreq} [MHz]")
        print(f"bw: {self.this_par.bw} [MHz]")
        fcorrected = 0.5 * np.sqrt(self.this_par.bw**2 + 4*self.this_par.cfreq**2)

        # https://academic.oup.com/mnras/article/462/3/2587/2589386
        print(f"Corrected f (Geyer. M & Karastergiou. A., 2016): {fcorrected} [MHz]")

        # plot
        if self.show_plots or self.save_plots:
            fig = p.plot(xlabel = "Time [ms]", ylabel = "Flux Density (arb.)", show = False)
            ax = fig.axes[0]

            if self.plot_type == "lines":
                ax.plot(p.x, p.y, 'k--', zorder = 0, alpha = 0.4)

            # plot seperate gaussians
            tau = p.get_post_val()['tau']
            xmodl = np.linspace(p.x[0], p.x[-1], p.modelNpoints)
            for i in range(npulse):
                ai = p.get_post_val()[f'a{i+1}']
                sigi = p.get_post_val()[f'sig{i+1}']
                mui = p.get_post_val()[f'mu{i+1}']
                ax.plot(xmodl, scatt_pulse_profile(xmodl, a1 = ai, sig1 = sigi, mu1 = mui, tau = tau),
                        label = f"p{i+1}")
            ax.legend()

            if self.save_plots:
                if filename is None:
                    filename = f"{self.par.name}_fit_tscatt.png"
                
                plt.savefig(filename)
            
            if self.show_plots:
                plt.show()

        # save instance parameters
        self._save_new_params()

        return p






    def fit_depol(self, modified: bool = True, sigrm_max: float = 100.0, redo: bool = False,
                    fit_params: dict = None, filename: str = None, stk_sigma: float = 3.0,
                    stk_debias: bool = True, **kwargs):
        """
        Fit sigma_rm and p using either the modified or un-modified burns law for spectral
        depolarisation.

        parameters
        ----------
        modified : bool
            Use modified burns law (fit for sigma_rm and p). If set to False, the priors for p are set to [0.999, 1.0], by default true
        sigrm_max : float
            Maximum sigma_rm value to fit for, by default 100.0, 0.0 is the minimum. 
        redo : bool
            Reset fitting (remove cached files), by default false
        fit_params : dict
            Fitting parameters for (Bilby likelihood estimation only)
        stk_sigma : float
            Sigma threshold to use when calculating total polarisation fraction
        stk_debias : bool
            Debias Total polarisation fraction, by default true
        filename : str
            filename for saved figure image
        
        returns
        -------
        p : pyfit.fit
            fitting class with posteriors/results
        """

        log_title(f"Fitting for depolarisation using Bayesian method.", col = "lblue")
        ##====================##
        ## check if data valid##
        ##====================##

        # initilaise dicts
        fit_params = dict_init(fit_params)

        # init par
        self._load_new_params(**kwargs)

        # get data
        data = self.get_data(['fP'], get = True, stk_ratio = True, stk_sigma = stk_sigma, 
                                stk_debias = stk_debias, **kwargs)

        # pmin = 0.0
        static = {}
        if not modified:
            # pmin = 0.999
            static = {'p':1.0}


        # fit 
        p = fit(x = data['freq'], y = data['fP'], yerr = data['fPerr'], func = burnslaw, 
                method = "bayesian", prior = {'sig_rm':[0.0, sigrm_max], 'p':[0.0, 1.0]},
                static = static, fit_keywords = fit_params, residuals = self.residuals, 
                plotPosterior = self.plotPosterior)

        p.fit(redo = redo)

        # print results
        if self.verbose:
            p.stats()
        print(p)

        # plot
        if self.show_plots or self.save_plots:
            fig, ax = plt.subplots(1, 1, figsize = (8, 8))
            plot(x = p.x, y = p.y, yerr = p.yerr, ax = ax,
                    plot_type = self.plot_type, color = 'k', alpha = 0.7)
            wid = p.x[-1] - p.x[0]
            ax.plot(*p.get_model(x = np.linspace(p.x[0] - wid, p.x[-1] + wid)),
                    color = 'r', linestyle = '--', linewidth = 2.0, zorder = 0)
            ax.set_xlabel("Frequency [MHz]")
            ax.set_ylabel("p")
            fig.tight_layout()

            if self.save_plots:
                if filename is None:
                    filename = f"{self.par.name}_fit_depol.png"
                
                plt.savefig(filename)
            
            if self.show_plots:
                plt.show()

        # save instance parameters
        self._save_new_params()

        return p




    




    def calc_periodgram(self, **kwargs):
        """
        Calculate and return ACF of time series

        returns
        -------
        tIacf : 1D np.ndarray
            Time profile auto correlations
        tlabs : 1D np.ndarray
            Time lags [ms]
        
        """

        dat = self.get_data("tI", get = True, **kwargs)

        # get acf
        tIacf = acf(dat['tI'])

        # get time lag, but only from first non-zero time lag sample
        dt = dat['time'][1] - dat['time'][0]
        tlags = np.linspace(dt, dt * tIacf.size, tIacf.size)

        return tIacf, tlags



    def plot_periodgram(self, plot_log = False, filename = None, **kwargs):
        """
        Plot ACF of time series

        plot_log: bool
            Also plot log y-axis 

        """

        log_title(f"plotting periodgram [ACF]", col = "lblue")

        dat = self.get_data("tI", get = True, **kwargs)

        # get acf
        tIacf = acf(dat['tI'])

        # get time lag, but only from first non-zero time lag sample
        dt = dat['time'][1] - dat['time'][0]
        tlags = np.linspace(dt, dt * tIacf.size, tIacf.size)

        cols = 1
        if plot_log:
            cols = 2

        fig, ax = plt.subplots(1, cols, figsize = (6 * cols, 6))

        if plot_log:
            ax = ax.flatten()
        else:
            ax = [ax]
        
        # plot linear periodgram
        ax[0].plot(tlags, tIacf, 'k', linewidth = 1.5)
        ax[0].set_xlabel("Time lag [ms]", fontsize = 16)
        ax[0].set_ylabel("ACF power (arb.)", fontsize = 16)

        if plot_log:
            ax[1].plot(tlags, tIacf, 'k', linewidth = 1.5)
            ax[1].set_xlabel("Time lag [ms]", fontsize = 16)
            ax[1].set_yscale('log')

        fig.subplots_adjust(hspace = 0, wspace = 0)
        fig.tight_layout()


        if self.save_plots:
            if filename is None:
                filename = f"{self.par.name}_periodgram.png"
            else:
                filename += "_periodgram.png"
            
            plt.savefig(filename)
            
        
        if self.show_plots:
            plt.show()

        self._save_new_params()


        return fig




    def plot_subbands(self, N = 2, stk = "I", filename = None, **kwargs):
        """
        Plot multiple time series subbands

        Parameters
        ----------
        N : int
            Number of subbands to split up, by default 2
        stk : str
            Stokes parameter to plot, by default 'I'
        
        """

        log_title(f"Plotting stokes [{stk}] subbands", col = "lblue")

        self._load_new_params(**kwargs)

        if stk not in "IQUV":
            ValueError(f"stk = {stk}, is not a valid stokes parameter to plot")

        
        # TODO: plot in different axes for now, figure out way to plot on pulsetrain later
        fig, ax = plt.subplots(N, 1, figsize = (10, 2*N), sharex = True)
        ax = ax.flatten()

        # split into subbands
        freq_c = np.linspace(self.this_par.f_lim[0] + self.this_par.bw/2/N,           # freq centers
                         self.this_par.f_lim[1] - self.this_par.bw/2/N, N)
        freq_bins = np.linspace(*self.this_par.f_lim, N+1) 

        # get data
        for i in range(N):
            kwargs['f_crop'] = [freq_bins[i], freq_bins[i+1]]
            data = self.get_data("tI", get = True, **kwargs)
            plot_data(data, "tI", ax = ax[i], plot_type = self.plot_type)
            if i < N-1:
                ax[i].get_xaxis().set_visible(False)
            ax[i].scatter([],[], label = f"[{freq_bins[i]:.2f} - {freq_bins[i+1]:.2f}] MHz")
            ax[i].legend()
        
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0)

        if self.save_plots:
            if filename is None:
                filename = f"{self.par.name}_subbands.png"
            plt.savefig(filename)

        if self.show_plots:
            plt.show()

        self._save_new_params()

        return fig



    def plot_poincare(self, stk_type = "f", stk_sigma = 2.0, plot_data = True,
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
        stk_sigma : float, optional
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

        stk_i, stk_m = plot_poincare_track(pdat, ax, sigma = stk_sigma,
                    plot_data = plot_data, plot_model = plot_model, normalise = normalise,
                    n = n)
                
        # enlarge figure
        fig.tight_layout()
                    
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

        return fig













    def fit_RM(self, method = "RMquad", rm_prior: list = [-1000, 1000], 
                pa0_prior: list = [-3.1415926/2, 3.1415926/2], Inorm: bool = True, unwrap: bool = False, fit_params: dict = None, filename: str = None, 
                sigma: float = None, **kwargs):
        """
        Fit Spectra for Rotation Measure

        Parameters
        ----------
        method : str, optional
            Method to perform Rotation Measure fitting, by default "RMquad" \n
            [RMquad] - Fit for the Rotation Measure using the standard quadratic method \n
            [RMsynth] - Use the RM-tools RM-Synthesis method \n
            [QUfit] - Fit log-likelihood model of Stokes Q and U parameters (see bannister et al 2019 - supplementary)
        rm_prior : list
            priors for rotation measure, used in [QUfit], by default [-1000, 1000]
        pa0_prior : list
            priors for PA0, used in [QUfit], by default [-pi/2, pi/2] (Shouldn't need to change)
        Inorm: bool
            If True, will Normalize Stokes parameters by Stokes I when performing RMsynthesis ("RMsynth"), mainly used to bypass poor Stokes I modelling in RMTools library, by default True
        fit_params : dict, optional
            keyword parameters for fitting method, by default None \n
            [RMquad] - Scipy.optimise.curve_fit keyword params \n
            [RMsynth] - RMtools_1D.run_synth keyword params \n
            [QUfit] - bilby.run_sampler keyword params
        sigma : float, optional
            apply a sigma threshold to frequency data and mask before fitting RM, by default None
        filename : str, optional
            filename to save figure to, by default None

        Returns
        -------
        p : pyfit.fit
            pyfit class fitting structure
        rmDict: dict
            Dictionary of fitted values and data \n
            rm [float] rotation measure \n
            rm_err [float] error in rotation measure \n
            pa0 [float] position angle at f0 \n
            pa0_err [float] position angle err \n
            f0 [float] reference frequency at weighted mid-band \n
            phiArr [np.ndarray] Array of phi values (If RMsynth enabled) \n
            rmArr [np.ndarray] Array of faraday depth values (If RMsynth enabled) \n
            pa [np.ndarray] Array of Polarisation angles [Rad] \n
            pa_err [np.ndarray] Array of Polarisation angle errors [Rad]
        """
        log_title(f"Fitting for RM using [{method}] method.", col = "lblue")

        fit_params = dict_init(fit_params)
        self._load_new_params(**kwargs)

        # check which data products are needed
        if method in ["RMsynth", "RMquad", "QUfit"]:
            data_list = ["fI", "fQ", "fU"]
        
        else:
            log("Invalid method for estimating RM", stype = "err", lpf_col = self.pcol)
            return None, None
            
        if self.this_metapar.terr_crop is None:
            log("Must specify 'terr_crop' for rms crop if you want to use RMsynth or RMquad", stype = "err",
                lpf_col = self.pcol)
            return None, None

        if sigma is not None:
            # zap
            log("rewriting kwargs[zapchan] using sigma threshold only for RM fitting", lpf_col = self.pcol)
            zapstr = self.zap_channels(zapsigma = sigma, stDev = 0, overwrite = False)
            kwargs['zapchan'] = zapstr
        
        ## get data ##
        self.get_data(data_list, ignore_nans = True, **kwargs)
        if not self._isdata():
            return None, None

        # ## mask data based on S/N threshold given
        # if sigma is not None:
        #     mask = self._f["I"] > self._f['Ierr'] * sigma
        # else:
        #     mask = np.ones(self._f['I'].size, dtype = bool)

        ## run fitting for RM ##
        if method == "RMquad":
            # run quadrature method
            if self.this_par.f0 is None:
                log("f0 not given, using middle of band", stype = "warn", lpf_col = self.pcol)
                self.this_par.f0 = self.this_par.cfreq

            f0 = self.this_par.f0
            # run fitting
            rmDict = fit_RMquad(self._f['Q'], self._f['U'], self._f['Qerr'],
                                                  self._f['Uerr'], self._freq, f0, **fit_params)
            rmDict['f0'] = f0


        elif method == "RMsynth":
            # run RM synthesis method
            pa0_err = 0.0       # do this for now, TODO
            I, Q, U = self._f['I'], self._f['Q'], self._f['U']
            Ierr, Qerr, Uerr = self._f['Ierr'], self._f['Qerr'], self._f['Uerr'] 
            rmDict = fit_RMsynth(I, Q, U, Ierr, 
                                Qerr, Uerr, self._freq, Inorm = Inorm, **fit_params)
            rmDict['pa0_err'] = pa0_err

        
        elif method == "QUfit":

            # TODO: make reference frequency same as FDF?

            # run log-likelihood estimating for Q and U paramters
            f0 = self.par.cfreq
            Q, U = self._f['Q'], self._f['U']
            Ierr, Qerr, Uerr = self._f['Ierr'], self._f['Qerr'], self._f['Uerr']
            rmDict = RM_QUfit(Q = Q, U = U, Ierr = Ierr, Qerr = Qerr, 
                                                Uerr = Uerr, f = self._freq, rm_priors = rm_prior, 
                                                pa0_priors = pa0_prior, **fit_params)
            rmDict['f0'] = f0


        # function for plotting diagnostics
        # if method == "QUfit":
        #     def rmquad(f, rm, pa0):
        #         angs = pa0 + rm*c**2/(f*1e6)**2
        #         return 90/np.pi*np.arctan2(np.sin(2*angs), np.cos(2*angs))
        # else:
        def rmquad(f, rm, pa0):
            angs = pa0 + rm*c**2/1e12*(1/f**2 - 1/rmDict['f0']**2)
            return 90/np.pi*np.arctan2(np.sin(2*angs), np.cos(2*angs))
        
        def rmquad_unwrapped(f, rm, pa0):
            angs = pa0 + rm*c**2/1e12*(1/f**2)
            return 180/np.pi * angs


        # put into pyfit structure
        PA, PA_err = calc_PA(self._f['Q'], self._f['U'], self._f['Qerr'], self._f['Uerr'])

        rmDict['pa'] = PA.copy()
        rmDict['pa_err'] = PA_err.copy()

        p = fit(x = self._freq, y = 180/np.pi*PA, yerr = 180/np.pi*PA_err, func = rmquad,
                 residuals = self.residuals)
        p.set_posterior('rm', rmDict['rm'], rmDict['rm_err'], rmDict['rm_err'])
        p.set_posterior('pa0', rmDict['pa0'], rmDict['pa0_err'], rmDict['pa0_err'])
        p.set_posterior('f0', rmDict['f0'], 0.0, 0.0)

        # set values to fitted_params 
        self.fitted_params['RM'] = p.get_posteriors()
        self.fitted_params['RM']['f0'] = _posterior(rmDict['f0'], 0, 0)
        p._is_fit = True
        p._is_stats = True
        p._get_stats()
        print(p)

        # plot
        if self.save_plots or self.show_plots:

            pa_ylim = [-90, 90]
            if unwrap:
                y_wrap = p.y.copy()
                func_wrap = p.func
                y_unwrap, _, _ = unwrap_pa(p.x, p.y, rmDict['rm'], rmDict['pa0'])
                p.set(y = y_unwrap, func = rmquad_unwrapped)
                y_height = np.max(y_unwrap) - np.min(y_unwrap)
                pa_ylim = [np.min(y_unwrap) - 0.15*y_height, np.max(y_unwrap) + 0.15*y_height]
            
            fig = p.plot(xlabel = "Frequency [MHz]", ylabel = "PA [deg]", ylim = pa_ylim, show = False)

            if unwrap:
                p.set(y = y_wrap, func = func_wrap)
                if self.residuals:
                    fig.axes[1].set(ylim = [-360, 360])


            if self.save_plots:
                if filename is None:
                    filename = f"{self.par.name}_RM_fit.png"
                
                plt.savefig(filename)

            if self.show_plots:
                plt.show()


        self._save_new_params()


        return p, rmDict








    




    def plot_PA(self, Ldebias_threshold = 2.0, stk2plot = "ILV", flipPA = False, stk_ratio = False,
                stk_sigma = 3.0, stk_debias = False, fit_params: dict = None, filename: str = None, 
                save_files = False, **kwargs):
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
        stk_debias : bool, optional
            Plot stokes L and/or P debias, by default False
        stk_sigma : float, optional
            sigma threshold for error masking, data that is I < sigma * Ierr, mask it out or
            else weird overflow behavior might be present when calculating stokes ratios, by default 2.0
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

        # Get data
        if 't_crop' not in kwargs.keys():
            kwargs['t_crop'] = self.metapar.t_crop.copy()
        tcrop = kwargs['t_crop'].copy()
        kwargs['t_crop'] = self.get_ptcrop(tcrop)

        self.get_data(data_list, **kwargs)
        if not self._isdata():
            return None

        kwargs['t_crop'] = tcrop.copy()

        ## calculate PA
        stk_data = {"tQ":self._t["Q"], "tU":self._t["U"], "tQerr":self._t["Qerr"],
                    "tUerr":self._t["Uerr"], "tIerr":self._t["Ierr"]}
        PA, PA_err = calc_PAdebiased(stk_data, Ldebias_threshold = Ldebias_threshold)

        # create figure
        fig, AX = plt.subplot_mosaic("P;S;D", figsize = (12, 10), 
                    gridspec_kw={"height_ratios": [1, 2, 2]}, sharex=True)

        _x = np.linspace(*self.this_par.t_lim, PA.size)

        ## plot PA
        if np.any(~np.isnan(PA)) and np.any(~np.isnan(PA_err)):
            plot_PA(_x, PA, PA_err, ax = AX['P'], flipPA = flipPA)

        ## plot Spectra
        pdat = {'time':_x}
        for S in "IQUV":
            pdat[f"t{S}"] = self._t[S]
            pdat[f"t{S}err"] = self._t[f"{S}err"]

        plot_stokes(pdat, ax = AX['S'], stk_type = "t", stk2plot = stk2plot, Ldebias = stk_debias, 
                    plot_type = self.plot_type, stk_ratio = stk_ratio, sigma = stk_sigma)

        # plot t_crop
        ylim = AX['S'].get_ylim()
        AX['S'].fill_between(tcrop, *ylim, color = "coral", 
                                            zorder = 0, alpha = 0.15)
        AX['S'].set_ylim(ylim)

        ## plot dynamic spectra
        ds_freq_lims = fix_ds_freq_lims(self.this_par.f_lim, self.this_par.df)
        plot_dynspec(self._ds['I'], ax = AX['D'], aspect = 'auto', 
                       extent = [*self.this_par.t_lim,*ds_freq_lims], showzaps = self.show_dynzaps)
        AX['D'].set_ylabel("Frequency [MHz]")
        AX['D'].set_xlabel("Time [ms]")

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
                                                



    def calc_polfracs(self, stk_debias = False, peak_sigma = 3.0, peak_average_factor = 1, polprint = True, **kwargs):
        """
        Calculate polarisation fractions using a number of different methods.

        Parameters
        ----------
        debias : bool, optional
            Debiases Stokes L, P and abs(V), abs(Q) and abs(U), by default False.
        peak_sigma : float, optional
            Provide a threshold in terms of I/Ierr that will be used to mask the data
            before estimating the peak fraction of each stokes parameter. This will be 
            nessesary to filter out noisy data.
        peak_average_factor, float
            averaging (downsampling) factor to apply to X(t) stokes profiles to help estimate their peaks
        print: bool
            print polarisation fractions, by default True
        """

        def pol_print(_str):
            if polprint:
                print(_str)

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
        S = self.get_data(["tI", "tQ", "tU", "tV", "tL", "tP"], get = True, stk_debias = stk_debias, **kwargs)
        nsamp = S['tI'].size

        # check if error was given, else turn off debias
        err = True
        if self.this_metapar.terr_crop is None:
            stk_debias = False
            err = False
            log("No off-pulse crop given to calculate dibased L, P and/or |U/Q/V|, specify [terr_crop]...", stype = "warn")
            log("No peak fractions we be calculatedm specify [terr_crop]...", stype = "warn")
        
        # mask any '0.0' values for Debiasing
        for d in ['tL', 'tP']:
            mask = S[d] == 0.0
            S[d][mask] = np.nan
            if err:
                S[f"{d}err"][mask] = np.nan

        
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
            
            # # diagnostic plots
            # if self.show_plots or self.save_plots:
            #     fig, ax = plt.subplots(1, 1, figsize = (12,8))
                
            #     for s in "quvlp":
            #         _PLOT(S['time'], stk_frac[s], stk_frac[f'{s}err'], ax = ax, plot_type = self.plot_type, 
            #                 color = _G.stk_colors[s.upper()])
            #     ylim = ax.get_ylim()
            #     for s in "quvlp":
            #         # plot marker
            #         ax.plot([peaks_pos[s]]*2, ylim, color = _G.stk_colors[s.upper()], linestyle = "--",
            #                     label = f'${s}_{{peak}}$ at t = {peaks_pos[s]:.2f} ms')
                
            #     ax.set(ylim = ylim, xlabel = "Time [ms]", ylabel = "Stokes X/I fraction")
            #     ax.legend()

                # # save figure
                # if self.save_plots:
                #     filename = f"{self.par.name}_peak_polfracs.png"
                
                #     plt.savefig(filename)



        # Now we can print everything out
        debias_flag = "FALSE\n"
        if stk_debias:
            debias_flag = "TRUE\n"
        pol_print("\nStokes fractions:")
        pol_print("="*50)
        pol_print(f"debiased = " + debias_flag)

        def _print_err(val):
            if err:
                return f"{val:.4f}"
            else:
                return "None"


        pol_print("=======  CONTINUUM-ADDED Stokes fractions  =======\n")
        pol_print("These fractions are calculated by first")
        pol_print("integrating over the debiased polarisation profile")
        pol_print("="*50 + "\n")
        pol_print("Legend:")
        pol_print("l = sum(L(t))/sum(I(t))\n")
        pol_print("|q|".ljust(15) + f"{fracS['|q|']:.4f} +/- {_print_err(fracS['|q|err'])}")
        pol_print("|u|".ljust(15) + f"{fracS['|u|']:.4f} +/- {_print_err(fracS['|u|err'])}")
        pol_print("|v|".ljust(15) + f"{fracS['|v|']:.4f} +/- {_print_err(fracS['|v|err'])}")
        pol_print("l".ljust(15) + f"{fracS['l']:.4f} +/- {_print_err(fracS['lerr'])}")
        pol_print("p".ljust(15) + f"{fracS['p']:.4f} +/- {_print_err(fracS['perr'])}\n")
        
        pol_print("Integrated (Signed) Stokes Paramters")
        pol_print("q".ljust(15) + f"{fracS['q']:.4f}".ljust(7) + f" +/- {_print_err(fracS['qerr'])}")
        pol_print("u".ljust(15) + f"{fracS['u']:.4f}".ljust(7) + f" +/- {_print_err(fracS['uerr'])}")
        pol_print("v".ljust(15) + f"{fracS['v']:.4f}".ljust(7) + f" +/- {_print_err(fracS['verr'])}\n")

        pol_print("====  Vector-addded Stokes L and P fractions  ====")
        pol_print("="*50 + "\n")

        pol_print("Legend:")
        pol_print("l* = sqrt(q^2 + u^2)")
        pol_print("|l|* = sqrt(|q|^2 + |u|^2)")
        pol_print("p* = sqrt(q^2 + u^2 + v^2)")
        pol_print("|p|* = sqrt(|q|^2 + |u|^2 + |v|^2)")
        pol_print("p# = sqrt(l*^2 + |v|^2) = sqrt(q^2 + u^2 + |v|^2)\n")


        # calculate l* and p*
        fracS['l*'], fracS['l*err'] = calc_L(fracS['q'], fracS['u'], fracS['qerr'], fracS['uerr'])
        fracS['p*'], fracS['p*err'] = calc_P(fracS['q'], fracS['u'], fracS['v'], fracS['qerr'],
                                    fracS['uerr'], fracS['verr'])
        pol_print("l*".ljust(15) +  f"{fracS['l*']:.4f} +/- {_print_err(fracS['l*err'])}")
        pol_print("p*".ljust(15) + f"{fracS['p*']:.4f} +/- {_print_err(fracS['p*err'])}")

        # calculate |l|* and |p|*
        fracS['|l|*'], fracS['|l|*err'] = calc_L(fracS['|q|'], fracS['|u|'], fracS['|q|err'],
                                        fracS['|u|err'])
        fracS['|p|*'], fracS['|p|*err'] = calc_P(fracS['|q|'], fracS['|u|'], fracS['|v|'],
                                        fracS['|q|err'], fracS['|u|err'], fracS['|v|err'])
        fracS['p#'], fracS['p#err'] = calc_P(fracS['q'], fracS['u'], fracS['|v|'],
                                        fracS['qerr'], fracS['uerr'], fracS['|v|err'])

        pol_print("|l|*".ljust(15) + f"{fracS['|l|*']:.4f} +/- {_print_err(fracS['|l|*err'])}")
        pol_print("|p|*".ljust(15) + f"{fracS['|p|*']:.4f} +/- {_print_err(fracS['|p|*err'])}\n")
        pol_print("p#".ljust(15) + f"{fracS['p#']:.4f} +/- {_print_err(fracS['p#err'])}\n")

        pol_print("= Total polarisation fraction calculated L and V =")
        pol_print("="*50 + "\n")
        pol_print("Legend:")
        pol_print("p^ = sqrt(l^2 + v^2)")
        pol_print("|p|^ = sqrt(l^2 + |v|^2)\n")

        fracS['p^'], fracS['p^err'] = calc_L(fracS['l'], fracS['v'], fracS['lerr'], fracS['verr'])
        fracS['|p|^'], fracS['|p|^err'] = calc_L(fracS['l'], fracS['|v|'], fracS['lerr'], fracS['|v|err'])
        pol_print("p^".ljust(15) + f"{fracS['p^']:.4f} +/- {_print_err(fracS['p^err'])}")
        pol_print("|p|^".ljust(15) + f"{fracS['|p|^']:.4f} +/- {_print_err(fracS['|p|^err'])}\n")

        if err and polprint:
            print(f"= Peak absolute polarisation fraction at dt = [{self.this_par.dt * 1000:.0f}] us =")
            print(f"="*50, "\n")
            for s in "quvlp":
                print(f"{s}_peak".ljust(15), f"{peaks[s]:.4f} +/- {peaks[f'{s}err']:.4f}".ljust(25), f"at time t = {peaks_pos[s]:.2f} ms")
            if self.save_plots:
                print(f"\nPrinting out diagnostic plot of stokes polarisation fractions as a function of time [{filename}]\n")

        # if self.show_plots and err:
        #     plt.show()
        

        return fracS





    def reset_crop(self):
        """
        Reset crop parameters
        
        """

        self.set(t_ref = 0, t_crop = ['min', 'max'], terr_crop = None, 
                f_crop = ['min', 'max'])

        return
