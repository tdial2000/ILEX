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
from .pyfit import fit
import yaml

## import utils ##
from .utils import (load_data, save_data, dict_get,
                    dict_init, dict_isall,
                    merge_dicts, dict_null, get_stk_from_datalist, load_param_file)

from .data import *

## import FRB stats ##
from .fitting import (fit_RMquad, fit_RMsynth, lorentz,
                     make_scatt_pulse_profile_func)

## import FRB params ##
from .par import FRB_params, FRB_metaparams

# ## import FRB htr functions ##
# from .htr import make_stokes

## import globals ##
from .globals import _G, c

## import plot functions ##
from .plot import (plot_RM, plot_PA, plot_stokes,      
                  plot_poincare_track, create_poincare_sphere, plot_data)

## import processing functions ##
# from .FRBproc import proc_data

from .logging import log, get_verbose, set_verbose


from .master_proc import master_proc_data



    

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
    RM: float 
        Rotation Measure
    f0: float 
        Reference Frequency
    pa0: float 
        Position angle at reference frequency f0
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
    force: bool 
        En-force use of any loaded cropped data regardless of parameter differences
    savefig: bool 
        Save all created figures to files
    pcol: str 
        Color of text for logging
    empty: bool 
        Variable used to initialise FRB instance and data loading
    plot_err_type: str
        type of error to plot \n
        [lines] - scatter plot with error bars \n
        [regions] - patch regions surrounding line plot
    residuals: bool
        if true, a residual panel will appear when plotting a fit using pyfit, default is True
    plotPosterior: bool
        if true, will save posterior corner plot when fitting using bayesian method, default is True
    apply_tw: bool
        if true, apply time dependant weights when scrunching in time, i.e making spectra, default is True
    apply_fw: bool
        if true, apply freq dependant weights when scrunching in freq, i.e. making time profiles, default is True
    """




    ## [ INITIALISE FRB ] ##
    def __init__(self, name: str = _G.p['name'],    RA: str = _G.p['RA'],    DEC: str = _G.p['DEC'], 
                       DM: float = _G.p['DM'],      bw: int = _G.p['bw'],    cfreq: float = _G.p['cfreq'], 
                       t_crop = None,               f_crop = None,           tN: int = 1,
                       fN: int = 1,                 t_lim = _G.p['t_lim'],   f_lim = _G.p['f_lim'],
                       RM: float = _G.p['RM'],      f0: float = _G.p['f0'],  pa0: float = _G.p['pa0'],
                       verbose: bool = _G.hp['verbose'], norm = _G.mp['norm'], terr_crop = None, yaml_file = None):
        """
        Create FRB instance
        """
        
        self.par = FRB_params(name = name, RA = RA, DEC = DEC, 
                              DM = DM, bw = bw, cfreq = cfreq,
                              t_lim = t_lim, f_lim = f_lim, 
                              RM = RM, f0 = f0, pa0 = pa0)

        self.par.set_par(dt = 1e-3, df = 1)

        self.this_par = self.par.copy()
        self.prev_par = FRB_params(EMPTY = True)

        self.metapar = FRB_metaparams(t_crop = t_crop, f_crop = f_crop,
                        terr_crop = terr_crop, tN = tN, fN = fN, norm = norm)


        if t_crop is None:
            self.metapar.t_crop = [0.0, 1.0]    # crop of time axis
        if f_crop is None:
            self.metapar.f_crop = [0.0, 1.0]    # crop of frequency axis
            
        self.this_metapar = self.metapar.copy()
        self.prev_metapar = FRB_metaparams(EMPTY = True)


        ## Create data containers
        self.ds = {}                    # container for Dynamic spectra
        self.pol = {}                   # container for polarisation time series data (X, Y)

        for S in "IQUV":
            self.ds[S] = None
        
        for P in "XY":
            self.pol[P] = None

        
        ## data instance containers
        self._t = {}                    # container to store time series data
        self._f = {}                    # container to store frequency spectra data
        self._ds = {}                   # container to store dynamic spectra data
        self._freq = {}                 # container to store baseband frequency data
        self._time = {}                 # container to store time samples 

        # initilise data containers
        for S in "IQUV":
            self._t[S] = None
            self._t[f"{S}err"] = None
            self._f[S] = None
            self._f[f"{S}err"] = None
            self._ds[S]= None
        
        for P in "XY":
            self.pol[P]= None
  
        self.empty = True               # used to initialise FRB instance and data loading 
        self.verbose = verbose          # TODO: implement
        # set verbose
        set_verbose(self.verbose)

        self.force = False              # Force use of Parameters
        self.savefig = False            # save figures instead of plotting them

        self.pcol = 'cyan'              # color for verbose printing
        self.plot_err_type = "lines"    # type of errorbar plot
        self.residuals = False          # plot residuals when plotting fits
        self.plotPosterior = True      # plot posterior corner plot when plotting fits

        # weightings
        self.apply_tW = True                  # apply time weights
        self.apply_fW = True                  # apply freq weights

        self._isinstance = False        # if data instance is valid


        # quick load yaml file
        if yaml_file is not None:
            self.load_data(yaml_file = yaml_file)



    ##===============================================##
    ##            retrive data funtions              ##
    ##===============================================##

    
    ## [ LOAD IN DATA ] ##
    def load_data(self, ds_I: str = None, ds_Q: str = None, ds_U: str = None, ds_V: str = None,
                  pol_X: str = None, pol_Y: str = None, yaml_file: str = None, mmap = True, _init = False):
        """
        Load Stokes HTR data

        Parameters
        ----------
        ds_I: str 
            Filename of stokes I dynamic spectra
        ds_Q: str 
            Filename of stokes Q dynamic spectra
        ds_U: str 
            Filename of stokes U dynamic spectra
        ds_V: str 
            Filename of stokes V dynamic spectra
        pol_X: str 
            Filename of X polarisation time series
        pol_Y: str 
            Filename of Y polarisation time series
        yaml_file: str 
            parameter yaml file for FRB, default is None
        mmap: bool 
            Enable memory mapping for loading
        _init: bool 
            For initial Data loading
        """
        if yaml_file is not None:
            log("Loading from yaml file", lpf_col = self.pcol)

            # load pars
            yaml_pars = load_param_file(yaml_file)

            # extract pars
            pars = merge_dicts(yaml_pars['par'], yaml_pars['metapar'], yaml_pars['hyperpar'])
            # print(pars)
            # set pars
            self.set(**pars)

            # set weights if given
            self.par.set_weights(xtype = "t", **yaml_pars['weights']['time'])
            self.par.set_weights(xtype = "f", **yaml_pars['weights']['freq'])

            # set loaded files
            ds_I, ds_Q = yaml_pars['data']['dsI'], yaml_pars['data']['dsQ']
            ds_U, ds_V = yaml_pars['data']['dsU'], yaml_pars['data']['dsV']

            _init = True



        def init_par_from_load(x):
            """
            Initialise a number of parameters from loaded file
            """

            self.par.nchan = x.shape[0]                     # assumed that dyn spec is [freq,time]
            self.par.nsamp = x.shape[1]        
            # self.par.df    = self.par.bw / self.par.nchan           # delta freq in (MHz)
            # self.par.dt    = 1e-3 / self.par.df                     # delta time in (ms) [micro seconds]
            self.par.t_lim  = [0.0, self.par.dt * self.par.nsamp]

            self.empty = False



        ## dict. of files that will be loaded in
        data_files = {"ds_I": ds_I, "ds_Q": ds_Q, "ds_U": ds_U, "ds_V": ds_V,
                      "pol_X": pol_X, "pol_Y": pol_Y}


        # loop through files
        for key in data_files.keys():

            file = data_files[key]
            if file is not None:

                # load all dynamic spectra
                if "ds" in key:
                    self.ds[key[-1]] = load_data(file, mmap)
                    log(f"Loading stokes {key[-1]} Dynspec from: {file} with shape {self.ds[key[-1]].shape}",
                         lpf_col=self.pcol)

                    if self.empty or _init:
                        init_par_from_load(self.ds[key[-1]])

                # load all polarisation time series data
                elif "pol" in key:
                    self.pol[key[-1]] = load_data(file, mmap)
                    log(f"Loading Pol {key[-1]} from: {file} with shape {self.pol[key[-1]].shape}", 
                        lpf_col=self.pcol)



        
    ## [ SAVING FUNCTION - SAVE CROP OF DATA ] ##
    def save_data(self, data_list = None, name = None, **kwargs):
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

        log("Saving the following data products:", lpf_col = self.pcol)
        for data in data_list:
            log(f"[{data}]", lpf_col = self.pcol)

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

        # log(self.metapar.metapar2dict, lpf_col = self.pcol)
        # log(self.par, lpf_col = self.pcol)






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
    def get_data(self, data_list = "dsI", get = False, **kwargs):
        """
        Make new instance of loaded data. This will take a crop of the 
        loaded mmap-ed stokes data, pass it through the back-end processing
        function and save the data in memory in the ._ds, _t, _f and _freq
        class instance attributes.

        Parameters
        ----------
        data_list : List(str) or str, optional
            List of data products to load in, by default "dsI"
        get : bool, optional
            Return new crops of data, by default False and will only save data
            instances to class container attributes

        Returns
        -------
        data: Dict, optional
            Dictionary of processed data crops, by default None if get = False
        """
        # update par and metapar if nessesary
        self._load_new_params(**kwargs)

        

        # process data_list as str
        if type(data_list) == str:
            if data_list == "all":
                data_list = _G.hkeys[:-2]
            else:
                data_list = [data_list]
                
        log(f"Retrieving the following data: {data_list}", lpf_col = self.pcol)


        ## check if instance of data already exists
        if not self._check_instance(data_list):
            ## if check fails, will need to make new data
            # get the data_products needed to make new instance
            data_products = self._init_proc(data_list)


            ## first check if there is data to use
            if not self._isvalid(data_products):
                log("Loaded data not avaliable or incorrect DS shapes", stype = "err",
                    lpf_col = self.pcol)
                self._isinstance = False
                return 


            ## make new instances
            self._make_instance(data_list)


            ## set new instance param 
            self._save_new_params()
        
        else:
            log("Loading previous crop", lpf_col = self.pcol)

        self._isinstance = True

        # check if get is true
        if get:
            # return instance
            return self._get_instance(data_list)


        #return data
        return



    def _check_instance(self, data_list = None):
        """
        Run through unit checks of current instance of crops, do all crops match in shape,
        have any parameters changed since last call to get_data().

        Parameters
        ----------
        data_list : List(str), optional
            data to check, by default None

        Returns
        -------
        bool
            0 if failed, 1 if passed
        """

        if not dict_isall(self.prev_par.par2dict(), 
                          self.this_par.par2dict()) and not self.force:
            log("params do not match", stype = "warn")
            return 0
        
        if not dict_isall(self.prev_metapar.metapar2dict(),
                          self.this_metapar.metapar2dict()) and not self.force:
            log("metaparams do not match", stype = "warn")
            return 0

        # flags
        err_flag = self._iserr()

        
        ## check for data in instance
        _shape = {"ds":[], "t":[], "f":[]}
        freq_shape = 0
        time_shape = 0
        for data in data_list:
            stk = data[-1]
            
            # dynamic spectra
            if data[:2] == "ds":
                dat = self._ds[stk]
                typ = "ds"
            
            # time series
            if data[0] == "t":
                dat = self._t[stk]
                daterr = self._t[f"{stk}err"]
                typ = "t"

            # frequency spectra
            if data[0] == "f":
                dat = self._f[stk]
                daterr = self._f[f"{stk}err"]
                typ = "f"

            
            # check data
            if dat is None:
                log(f"{data} is missing, will be made", stype = "warn")
                return 0
            _shape[typ].append(dat.shape)
            if err_flag and typ != "ds":
                if daterr is None:
                    log(f"{data}err is missing, will be made", stype = "warn")
                    return 0
                if typ == "f":
                    _shape[typ].append(daterr.shape)


            # frequency band array
            elif data == "freq":
                if self._freq is None:
                    log("freq data is missing, will be made", stype = "warn")
                    return 0
                freq_shape = self._freq.shape
            
            elif data == "time":
                if self._time is None:
                    log("time data is missing, will be made", stype = "warn")
                    return 0
                time_shape = self._time.shape


        ## check if all shapes match up
        if not all(x==_shape['ds'][0] for x in _shape['ds']):
            log("cropped ds shapes do not match, will be remade", stype = "warn")
            return 0
        
        if not all(x==_shape['t'][0] for x in _shape['t']):
            log("cropped t shapes do not match, will be remade", stype = "warn")
            return 0

        if not all(x==_shape['f'][0] for x in _shape['f']):
            log("cropped f shapes do not match, will be remade", stype = "warn")
            return 0

        ## check if cross products i.e. ds and t match up
        if len(_shape['ds']) > 0 and len(_shape['t']) > 0:
            if _shape['ds'][0][1] != _shape['t'][0][0]:
                log(f"# samples for ds and t do not match, data will be remade\n{_shape['ds'][0][1]} != {_shape['t'][0][0]}",
                    stype = "warn")
                return 0
        
        if len(_shape['ds']) > 0 and len(_shape['f']) > 0:
            if _shape['ds'][0][0] != _shape['f'][0][0]:
                log(f"# channels for ds and f do not match, data will be remade\n{_shape['ds'][0][0]} != {_shape['t'][0][0]}",
                    stype = "warn")
                return 0
        
        if len(_shape['ds']) > 0 and freq_shape > 0:
            if _shape['ds'][0][0] != freq_shape:
                log(f"# channels for ds and freq do not match, data will be remade\n{_shape['ds'][0][0]} != {freq_shape}",
                    stype = "warn")
                return 0

        if len(_shape['ds']) > 0 and time_shape > 0:
            if _shape['ds'][0][1] != time_shape:
                log(f"# samples for ds and time do not match, data will be remade\n{_shape['ds'][0][1]} != {time_shape}",
                    stype = "warn")
                return 0
            
        if len(_shape['f']) > 0 and freq_shape > 0:
            if _shape['f'][0][0] != freq_shape:
                log(f"# channels for f and freq do not match, data will be remade\n{_shape['f'][0][0]} != {freq_shape}",
                    stype = "warn")
                return 0
        
        if len(_shape['t']) > 0 and time_shape > 0:
            if _shape['t'][0][0] != time_shape:
                log(f"# samples for t and time do not match, data will be remade\n{_shape['t'][0][0]} != {time_shape}",
                    stype = "warn") 
                return 0               

        
        ## all checks passed, return true
        return 1



    def _get_instance(self, data_list = None):
        """
        Grap current instance of crops

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

        # flags
        err_flag = self._iserr()

        for data in data_list:
            stk = data[-1]
            # dynamic spectra
            if "ds" in data:
                new_data[data] = self._ds[stk].copy()

            # time series
            elif "t" in data:
                new_data[data] = self._t[stk].copy()
                if err_flag:
                    new_data[f"{data}err"] = self._t[f"{stk}err"]

            # frequency spectra
            elif "f" in data:
                new_data[data] = self._f[stk].copy()
                if err_flag:
                    new_data[f"{data}err"] = self._f[f"{stk}err"]

        # also add freqs
        if self._freq is not None:
            new_data['freq'] = self._freq.copy()
        else:
            log("Couldn't get freq array, something went wrong", stype = "warn")

        # also add times
        if self._time is not None:
            new_data['time'] = self._time.copy()
        else:
            log("Couldn't get time array, something went wrong", stype = "warn")
        

        return new_data



    def _make_instance(self, data_list = None):
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


        # loop through data products
        freqs = self.par.get_freqs()
        

        # set up parameter dictionary
        full_par = merge_dicts(self.this_metapar.metapar2dict(), 
                                 self.this_par.par2dict())

        # get tW and fW
        temp_w_par = self.par.copy()
        temp_w_par.update_from_crop(t_crop = full_par['t_crop'],
                                        f_crop = full_par['f_crop'])

        if self.apply_tW:
            full_par['tW'] = self.par.tW.get_weights(x = temp_w_par.get_times())
        if self.apply_fW:
            full_par['fW'] = self.par.fW.get_weights(x = temp_w_par.get_freqs())

        # pass through to backend processing script
        _ds, _t, _f, self._freq = master_proc_data(self.ds, freqs, 
                                                            data_list, full_par)

        

        log("Saving new data products to latest instance", lpf_col = self.pcol)

        # dynspecs
        ds_list = _ds.keys()
        for key in ds_list:
            if _ds[key] is not None:
                self._ds[key] = _ds[key].copy()
                _ds[key] = None
        
        # time series
        t_list = _t.keys()
        aval_key_for_time = None
        for key in t_list:
            if _t[key] is not None: 
                if "err" not in key:
                    aval_key_for_time = key
                self._t[key] = _t[key].copy()
                _t[key] = None
        
        # freq spectra
        f_list = _f.keys()
        for key in f_list:
            if _f[key] is not None:
                self._f[key] = _f[key].copy()
                _f[key] = None

        if aval_key_for_time is not None:
            # set new time (self._time)
            self._time = np.linspace(*self.this_par.t_lim, self._t[aval_key_for_time].size, endpoint = False)

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



    
    def _init_proc(self, data_list):
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
        if (("Q" in stk) != ("U" in stk)) and self.this_par.RM != 0.0:
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
        # copy over current hyperparams to kwargs
        metapar = self.metapar.metapar2dict()
        kw = kwargs.keys()
        for key in metapar.keys():
            if key not in kw:
                kwargs[key] = metapar[key]

        # make sure metaparameters are updated first  
        self._proc_kwargs(**kwargs)
        
        # update self.this_metapar
        self._update_metapar(**kwargs)

        # update self.this_par
        self._update_par(**kwargs)

        # update hyper parameters
        self._update_hyperpar(**kwargs)





    

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

        def check_crop_for_str(crop, domain):
            """ Check for crop "min" and "max" specifiers"""
            if domain == "t":
                _vars = self.par.t_lim
            elif domain == "f":
                _vars = self.par.f_lim
            else:
                log("Something went wrong converting crops, no domain chosen.", stype = "err")

            phase_vars = [0.0, 1.0]

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

            if kwargs['t_crop'][0] > 1.0 or kwargs['t_crop'][1] > 1.0:
                prev_t = kwargs['t_crop'].copy()
                new_t,_ = self.par.lim2phase(t_lim = prev_t)
                kwargs['t_crop'][0], kwargs['t_crop'][1] = new_t[0], new_t[1]

                if kwargs['t_crop'][0] < 0.0: kwargs['t_crop'][0] = 0.0
                if kwargs['t_crop'][1] > 1.0: kwargs['t_crop'][1] = 1.0

                log(f"Converting Time crop {prev_t} ms -> {kwargs['t_crop']} phase units", lpf = False)

        # check if t_crop has been given in units of ms
        if "f_crop" in keys:

            kwargs['f_crop'][0], kwargs['f_crop'][1] = check_crop_for_str(kwargs['f_crop'], "f")

            if kwargs['f_crop'][0] > 1.0 or kwargs['f_crop'][1] > 1.0:
                prev_f = kwargs['f_crop'].copy()
                _, new_f = self.par.lim2phase(f_lim = prev_f)
                kwargs['f_crop'][0], kwargs['f_crop'][1] = new_f[0], new_f[1]

                if kwargs['f_crop'][0] < 0.0: kwargs['f_crop'][0] = 0.0
                if kwargs['f_crop'][1] > 1.0: kwargs['f_crop'][1] = 1.0

                log(f"Converting Freq crop {prev_f} MHz -> {kwargs['f_crop']} phase units", lpf = False)

        # check if terr_crop has been given in units of ms
        if "terr_crop" in keys:
            if kwargs['terr_crop'] is not None:
                
                kwargs["terr_crop"][0], kwargs["terr_crop"][1] = check_crop_for_str(kwargs["terr_crop"], "t")
                if kwargs['terr_crop'][0] > 1.0 or kwargs['terr_crop'][1] > 1.0:
                    prev_t = kwargs['terr_crop'].copy()
                    new_t,_ = self.par.lim2phase(t_lim = prev_t)
                    kwargs['terr_crop'][0], kwargs['terr_crop'][1] = new_t[0], new_t[1]

                    if kwargs['terr_crop'][0] < 0.0: kwargs['terr_crop'][0] = 0.0
                    if kwargs['terr_crop'][1] > 1.0: kwargs['terr_crop'][1] = 1.0

                    log(f"Converting err Time crop {prev_t} ms -> {kwargs['terr_crop']} phase units", lpf = False)
        
            



    




    





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
        outstr += "==============================\n"
        outstr += "==== FRB META Parameters =====\n"
        outstr += "==============================\n"
        outstr += "PARAM:       PROD:       INST:\n"
        outstr += "==============================\n"

        for _,key in enumerate(_G.mp.keys()):
            val = getattr(self.metapar, key)
            val2 = getattr(self.prev_metapar, key)
            
            outstr += f"{key}:  {val}   {val2}\n"


        outstr += "\n"
        outstr += "==============================\n"
        outstr += "==== FRB DATA Parameters =====\n"
        outstr += "==============================\n"
        outstr += "PARAM:       PROD:       INST:\n"
        outstr += "==============================\n"

        for _,key in enumerate(_G.p.keys()):
            val = getattr(self.par, key)
            val2 = getattr(self.prev_par, key)
            outstr += f"{key}:  {val}   {val2}\n"

        
        # outline loaded data
        outstr += "\n"
        outstr += "=========================\n"
        outstr += "==    DATA products    ==\n"
        outstr += "=========================\n"
        outstr += "TYPE:    SHAPE\n"
        outstr += "==============\n"
        for P in "XY":
            if self.pol[P] is not None:
                outstr += f"{P}:  {list(self.pol[P].shape)}\n"
        
        
        for S in "IQUV":
            if self.ds[S] is not None:
                outstr += f"{S}:   {list(self.ds[S].shape)}\n"
        
        #now print data instance
        outstr += "\n"
        outstr += "=========================\n"
        outstr += "==    DATA instance    ==\n"
        outstr += "=========================\n"
        outstr += "TYPE:    SHAPE\n"
        outstr += "==============\n"
        ds_str = ""
        t_str = "\n"
        f_str = "\n"
        for S in "IQUV":
            if self._ds[S] is not None:
                ds_str += f"ds{S}:   {list(self._ds[S].shape)}\n"
            
            if self._t[S] is not None:
                t_str += f"t{S}:    {list(self._t[S].shape)}\n"
            if self._t[f"{S}err"] is not None:
                Serr = f"{S}err"
                t_str += f"t{S}err: {self._t[Serr]}\n"
            
            if self._f[S] is not None:
                f_str += f"f{S}:    {list(self._f[S].shape)}\n"
            if self._f[f"{S}err"] is not None:
                Serr = f"{S}err"
                f_str += f"f{S}err: {list(self._f[Serr].shape)}\n"
        
        outstr += ds_str + t_str + f_str

        if self._freq is not None and len(self._freq) > 0:
            outstr += f"freqs: top:{self._freq[0]}, bottom:{self._freq[-1]}\n"

            


        return outstr



    
























    ##===============================================##
    ##            Further FRB processing             ##
    ##===============================================##


    ## [ FIND FRB PEAK AND TAKE REGION AROUND IT ] ##
    def find_frb(self, sigma: int = 5, _guard: float = 10, _width: float = 50, 
                _roughrms: float = 100, **kwargs):
        """
        This function searches the stokes I dynamic spectrum for the most likely
        location of the frb. It's important to note that this function will look through
        the entire dataset regardless of crop parameters. It will first scrunch, so if memory
        is an issue first set 'tN'.

        Parameters
        ----------
        sigma: int 
            S/N threshold
        _guard: float 
            gap between estiamted pulse region and 
            off-pulse region for rms and baseband estimation, in (ms)
        _width: float 
            width of off-pulse region on either side of pulse region in (ms)
        _roughrms: float 
            rough offset from peak on initial S/N threshold in (ms)
        **kwargs: 
            FRB parameters + FRB meta-parameters

        Returns
        -------
        t_crop: list
            New Phase start and end limits for found frb burst

        """
        log("Looking for FRB burst", lpf_col = self.pcol)

        ##====================##
        ## check if data valid##
        ##====================## 

        kwargs['t_crop'] = [0.0, 1.0]
        kwargs['f_crop'] = [0.0, 1.0]  

        # get dynI spectra (first scrunch)
        
        self.get_data("dsI", **kwargs)
        if not self._isdata():
            return None
        


        ##==========================##
        ## First (rough) FRB search ##
        ##==========================##
        log("Proceeding with First rough FRB search", lpf_col = self.pcol)





        # find max peak
        tpro = np.mean(self._ds['I'],axis = 0)
        peak = np.where(tpro == np.max(tpro))[0][0]                                          # peak sample
        peak /= self._ds['I'].shape[1]                                                       # peak phase


        # take 2 equal length regions away from the peak on either side
        # and measure rms as a rough first estimate
        pguard = _guard/(self.this_par.nsamp * self.this_par.dt)                                 # phase guard
        pwidth = _width/(self.this_par.nsamp * self.this_par.dt)                                 # phase width
        proughrms = _roughrms/(self.this_par.nsamp * self.this_par.dt) # phase rough rms


        rms_lhs = pslice(self._ds['I'], peak - proughrms - pwidth, peak - proughrms,1)       # lhs region
        rms_rhs = pslice(self._ds['I'], peak + proughrms, peak + proughrms + pwidth,1)       # rhs region

        rms_dyn = np.concatenate((rms_lhs,rms_rhs),axis = 1)                                 # full rms region
        rms_val = np.mean(np.mean(rms_dyn,axis = 0)**2)**0.5                                 # rms value


        # apply initial rough S/N estimate
        sig_i = np.where(tpro / rms_val > sigma)[0]                                          # where signal
                                                                                             # > sigma
        p_start = sig_i[0]/self._ds['I'].shape[1] - pguard - pwidth                          # new start
        p_end   = sig_i[-1]/self._ds['I'].shape[1] + pguard + pwidth                         # new end


        ##===============================##
        ## second (optomized) FRB search ##
        ##===============================##
        log("Proceeding with Optomised FRB search", lpf_col = self.pcol)


        # crop region
        dynI = pslice(self._ds['I'],p_start,p_end,1)                                         # crop new region


        # recalculate phase positions of rms
        pwidth = _width/(self.this_par.nsamp * self.this_par.dt)                                 # phase width

        rms_lhs = pslice(dynI,0.0,pwidth,1)
        rms_rhs = pslice(dynI,1.0 - pwidth,1.0,1)
        rms_dyn = np.concatenate((rms_lhs,rms_rhs),axis = 1)


        # S/N calculation, find where signal is greater than S/N = sigma
        rms_t = np.mean(rms_dyn, axis = 0) 
        rms_val = np.mean(rms_t**2)**0.5

        tpro = np.mean(dynI, axis = 0)                                                       # calculate S/N
        sig_i = np.where(tpro / rms_val > sigma)[0]


        # calculate new t_crop
        t_crop = [0.0,1.0]
        t_crop[0] = p_start + sig_i[0]/tpro.size*(p_end - p_start)
        t_crop[1] = p_start + sig_i[-1]/tpro.size*(p_end - p_start)
        width = t_crop[1] - t_crop[0]
        t_crop[0] -= 0.1*width
        t_crop[1] += 0.1*width

        self.metapar.set_metapar(t_crop = t_crop)

        log("New t_crop: [{:.5f}, {:.5f}]".format(t_crop[0],t_crop[1]), lpf_col = self.pcol)


        # clear dsI
        self._clear_instance(data_list = ["dsI"])

        return t_crop
    










    ##===============================================##
    ##                Plotting Methods               ##
    ##===============================================##
    def plot_data(self, data = "dsI", ax = None, filename: str = None, plot_err_type = "regions", **kwargs):
        """
        General Plotting function, choose to plot either dynamic spectrum or time series 
        data for all stokes parameters

        Parameters
        ----------
        data : str, optional
            type of data to plot, by default "dsI"
        filename : str, optional
            filename to save figure to, by default None
        plot_err_type : str, optional
            type of error plotting, by default "regions"\n
            [regions] - Show error in data as shaded regions
            [lines] - Show error in data as tics in markers

        Returns
        -------
        fig : figure
            Return Figure Instance
        """        

        log(f"plotting {data}", lpf_col = self.pcol)

        # get data
        pdat = self.get_data(data_list = data, get = True, **kwargs)
        if not self._isdata():
            return None

        stk = data[-1]

        if data[0] == "t":
            pdat['time'] = np.linspace(*self.this_par.t_lim, self._t[stk].size)
        else:
            pdat['time'] = np.array(self.this_par.t_lim)

        # plot 
        fig = plot_data(pdat, data, ax = ax, filename = filename, plot_err_type = plot_err_type)

        self._save_new_params()

        return fig












    def plot_stokes(self, ax = None, plot_L = False, Ldebias = False, debias_threshold = 2.0, 
            stk_type = "f", stk2plot = "IQUV", filename: str = None, **kwargs):
        """
        Plot Stokes data, by default stokes I, Q, U and V data is plotted

        Parameters
        ----------
        ax: _axes_
            matplotlib.pyplot.axes object to plot to, default is None
        plot_L : bool, optional
            Plot stokes L instead of Q and U, by default False
        Ldebias : bool, optional
            Plot stokes L debias, by default False
        debias_threshold : float, optional
            sigma threshold for error masking, data that is < debias_threshold * Ierr, mask it out or
            else weird overflow behavior might be present when calculating stokes ratios, by default 2.0
        stk_type : str, optional
            Type of stokes data to plot, "f" for Stokes Frequency data or "t" for time data, by default "f"
        stk2plot : str, optional
            string of stokes to plot, for example if "QV", only stokes Q and V are plotted, by default "IQUV"
        filename : str, optional
            name of file to save figure image, by default None
        **kwargs : Dict
            FRB parameter keywords

        Returns
        -------
        fig : figure
            Return figure instance

        """

        log(f"plotting stokes data", lpf_col = self.pcol)

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
        fig = plot_stokes(data, plot_L = plot_L, Ldebias = Ldebias, stk_type = stk_type,
                    debias_threshold = debias_threshold, stk2plot = stk2plot,
                    filename = filename, plot_err_type = self.plot_err_type, ax = ax) 
    

        self._save_new_params()

        return fig





    def plot_crop(self, stk = "I", **kwargs):
        """
        Plot current crop of of data along with off-pulse crop if given

        Parameters
        ----------
        stk : str, optional
            Stokes data to plot, by default "I"
        """

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
        print(tpad)

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

        # get data
        kwargs['t_crop'] = [*tcrop_ds]
        kwargs['f_crop'] = [*fcrop_ds]
        print(kwargs)
        self.get_data([f"ds{stk}"], **kwargs)



        # plot dynamic spectra
        fig = plt.figure(figsize = (12,12))
        plt.imshow(self._ds[stk], aspect = 'auto', extent = [*self.this_par.t_lim, 
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


        plt.show()





    ## [ PLOT LORENTZ OF CROP ] ##
    def fit_scintband(self, method = "bayesian",priors: dict = None, statics: dict = None, 
                     fit_params: dict = None, plot = False, redo = False, filename: str = None, **kwargs):
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
        plot : bool, optional
            plot data, by default False
        filename : str, optional
            Save figure to file, by default None

        Returns
        -------
        p: pyfit.fit
            pyfit class structure
        """
        
        log("Fitting for Scintillation bandwidth", lpf_col = self.pcol)
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
        
        # caculate acf of residuals
        y = acf(residuals(self._f['I']))
        # lags
        x = np.linspace(self.this_par.df, self.this_par.bw - self.this_par.df,
                         y.size)

        # get error in y
        yerr = None
        # if self._iserr():
        #     yerr = ccf(self._f['Ierr']**2, residuals(self._f['I'])**2)[1:]
        #     yerr = np.sqrt(yerr + yerr[::-1])



        # create instance of fitting
        p = fit(x = x, y = y, yerr = yerr, func = lorentz, prior = priors,
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
        
        if self.verbose:
            p.stats()
            print(p)

        ##===================##
        ##   do plotting     ##
        ##===================##

        
        
        if plot:
            p.plot(xlabel = "Freq [MHz]", ylabel = "Norm acf", filename = filename)

        # update instance par
        self._save_new_params()

        return p












    ## [ FIT SCATTERING TIMESCALE ] ##
    def fit_tscatt(self, method = "bayesian", npulse = 1, priors: dict = None, statics: dict = None, 
                   fit_params: dict = None, plot = False, redo = False, filename: str = None, **kwargs):
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
        plot : bool, optional
            plot data after modelling is complete, by default False
        redo : bool, optional
            if True, will redo fitting in the case that results are cached, this is mainly for BILBY fitting, default is False
        filename : str, optional
            filename to save final plot to, by default None

        Returns
        -------
        p: pyfit.fit
            pyfit class structure
        """        
        log("Fitting for Scattering Time", lpf_col = self.pcol)
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
        p = fit(x = x, y = y, yerr = err, func = make_scatt_pulse_profile_func(npulse),
                prior = priors, static = statics, fit_keywords = fit_params, method = method,
                residuals = self.residuals, plotPosterior = self.plotPosterior) 

        # fit 
        p.fit(redo = redo)
        
        # print best fit parameters
        if self.verbose:
            p.stats()
            print(p)

        # plot
        if plot:
            p.plot(xlabel = "Time [ms]", ylabel = "Flux Density (arb.)", filename = filename)

        # save instance parameters
        self._save_new_params()

        return p

    












    def plot_poincare(self, filename = None, stk_type = "f", sigma = 2.0, plot_data = True,
                        plot_model = False, n = 5, normalise = True, **kwargs):
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
        **kwargs : Dict
            FRB parameter keywords

        Returns
        -------
        fig : figure
            Return figure instance
        """    
        log("Plotting stokes data to poincare sphere", lpf_col = self.pcol)

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

        data = self.get_data(data_list = data_list, get = True, **kwargs)
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

        plot_poincare_track(pdat, ax, sigma = sigma, filename = filename,
                    plot_data = plot_data, plot_model = plot_model, normalise = normalise,
                    n = n)

        plt.show()

        self._save_new_params()

        return fig













    def fit_RM(self, method = "RMquad", plot = True, fit_params: dict = None, 
               filename: str = None, **kwargs):
        """
        Fit Spectra for Rotation Measure

        Parameters
        ----------
        method : str, optional
            Method to perform Rotation Measure fitting, by default "RMquad" \n
            [RMquad] - Fit for the Rotation Measure using the standard quadratic method \n
            [RMsynth] - Use the RM-tools RM-Synthesis method
        plot : bool, optional
            plot data after fitting if true, by default True
        fit_params : dict, optional
            keyword parameters for fitting method, by default None \n
            [RMquad] - Scipy.optimise.curve_fit keyword params \n
            [RMsynth] - RMtools_1D.run_synth keyword params
        filename : str, optional
            filename to save figure to, by default None

        Returns
        -------
        p : pyfit.fit
            pyfit class fitting structure
        """
        log(f"Retrieving RM", lpf_col = self.pcol)

        fit_params = dict_init(fit_params)
        self._load_new_params(**kwargs)

        # check which data products are needed
        if method == "RMquad":
            data_list = ["fQ", "fU"]

        elif method == "RMsynth":
            data_list = ["fI", "fQ", "fU"]
        
        else:
            log("Invalid method for estimating RM", stype = "err", lpf_col = self.pcol)
            return None
            
        if self.this_metapar.terr_crop is None:
            log("Must specify 'terr_crop' for rms crop if you want to use RMsynth or RMquad", stype = "err",
                lpf_col = self.pcol)
            return (None, ) * 5
        
        ## get data ##
        self.get_data(data_list, **kwargs)
        if not self._isdata():
            return None

        ## run fitting for RM ##
        if method == "RMquad":
            # run quadrature method
            if self.this_par.f0 is None:
                log("f0 not given, using middle of band", stype = "warn", lpf_col = self.pcol)
                self.this_par.f0 = self.this_par.cfreq

            f0 = self.this_par.f0
            # run fitting
            rm, rm_err, pa0, pa0_err = fit_RMquad(self._f['Q'], self._f['U'], self._f['Qerr'],
                                                  self._f['Uerr'], self._freq, f0, **fit_params)


        elif method == "RMsynth":
            # run RM synthesis method
            pa0_err = 0.0       # do this for now, TODO
            I, Q, U = self._f['I'], self._f['Q'], self._f['U']
            Ierr, Qerr, Uerr = self._f['Ierr'], self._f['Qerr'], self._f['Uerr'] 
            rm, rm_err, f0, pa0 = fit_RMsynth(I, Q, U, Ierr, Qerr, Uerr, self._freq,
                                          **fit_params)


        # put into pyfit structure
        PA, PA_err = calc_PA(self._f['Q'], self._f['U'], self._f['Qerr'], self._f['Uerr'])

        def rmquad(f, rm, pa0):
            angs = pa0 + rm*c**2/1e12*(1/f**2 - 1/f0**2)
            return 90/np.pi*np.arctan2(np.sin(2*angs), np.cos(2*angs))

        p = fit(x = self._freq, y = 180/np.pi*PA, yerr = 180/np.pi*PA_err, func = rmquad,
                 residuals = self.residuals)
        p.set_posterior('rm', rm, rm_err, rm_err)
        p.set_posterior('pa0', pa0, pa0_err, pa0_err)
        p.set_posterior('f0', f0, 0.0, 0.0)

        # set values to par class
        self.par.set_par(RM = rm, pa0 = pa0, f0 = f0)
        p._is_fit = True

        # plot
        if plot:
            
            p.plot(xlabel = "Frequency [MHz]", ylabel = "PA [deg]", ylim = [-90, 90],
            filename = filename)


        self._save_new_params()


        return p








    




    def plot_PA(self, Ldebias_threshold = 2.0, plot_L = False, flipPA = False,
                fit_params: dict = None, filename: str = None, **kwargs):
        """
        Plot Figure with PA profile, Stokes Time series data, and Stokes I dyspec. If RM is not 
        specified, will be fitted first.

        Parameters
        ----------
        Ldebias_threshold : float, optional
            Sigma threshold for PA masking, by default 2.0
        plot_L : bool, optional
            Plot Linear polarisation L stokes time series instead of U and Q, by default False
        flipPA : bool, optional
            Plot PA between [0, 180] degrees instead of [-90, 90], by default False
        fit_params : dict, optional
            keyword parameters for fitting method, by default None \n
            [RMquad] - Scipy.optimise.curve_fit keyword params \n
            [RMsynth] - RMtools_1D.run_synth keyword params
        filename : str, optional
            filename of figure to save to, by default None

        Returns
        -------
        p : pyfit.fit
            pyfit class fitting structure
        fig : figure
            Return figure instance
        """
        log("Plotting PA", lpf_col = self.pcol)

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

        plot_stokes(pdat, ax = AX['S'], stk_type = "t", plot_L = plot_L, Ldebias = True, 
                    plot_err_type = self.plot_err_type)

        ## plot dynamic spectra
        AX['D'].imshow(self._ds['I'], aspect = 'auto', 
                       extent = [*self.this_par.t_lim,*self.this_par.f_lim])
        AX['D'].set_ylabel("Frequency [MHz]", fontsize = 12)
        AX['D'].set_xlabel("Time [ms]", fontsize = 12)

        # plot peak in time
        peak = np.argmax(self._t['I'])
        x = np.linspace(*self.this_par.t_lim, self._t['I'].size)[peak]
        for A in "PDS":
            axylim = AX[A].get_ylim()
            AX[A].plot([x, x], axylim, 'r--')
            AX[A].set_ylim(axylim)

        # adjust figure
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0)
        AX['P'].get_xaxis().set_visible(False)
        AX['S'].get_xaxis().set_visible(False)

        if filename is not None:
            plt.savefig(filename)


        self._save_new_params()

        plt.show()

        return fig
                                                