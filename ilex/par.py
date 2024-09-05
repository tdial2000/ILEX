##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 11/10/2023 
##
##
## 
## 
## FRB params structure
## 
## 
##
##===============================================##
##===============================================##
# imports
import yaml
from copy import deepcopy
from .globals import _G
from .utils import dict_init
import numpy as np
import math
import inspect
from .fitting import *
from math import ceil

from .logging import log



class weights:
    """
    Weights structure, used for storing weights or weight functions that can be
    evaluated.


    Attributes
    ----------
    W: ndarray or array-like
        weights
    x: ndarray or array-like
        sample array defining weights, used for interpolation and function evaluation
    func: <lambda>
        function to evaluate to get weights
    method: str
        method to retrive weights \n
        [None] - Retrieve weights as is, either scalar or array \n
        [interp] - interpolate weights based on bounds and W (weights) \n
        [func] - Retrieve weights by evaluating function
    args: dict
        dictionary of arguments used in evaluating weights function
    norm: bool, optional
        Normalise weights, default is True
    
    
    """

    def __init__(self, W = None, x = None, func = None, method = None, args = None, norm = True):

        # initialise arguments dictionary
        args = dict_init(args)

        # set parameters to defaults first, then use the set method
        self.W = None
        self.x = None
        self.func = None
        self.method = "None"
        self.args = {}
        self.norm = None

        # set attributes whilst running some checks
        self._set(W = W, x = x, func = func, method = method, args = args, norm = norm)


    def set(self, **kwargs):
        """
        Set attributes for weights class

        """         

        # this block of code makes sure only the correct attributes are taken from the original inputs
        weights_kwargs = {}

        for key in kwargs.keys():
            if key in ["W", "x", "func", "method", "args", "norm"]:
                weights_kwargs[key] = deepcopy(kwargs[key])

        self._set(**weights_kwargs)
        

    def _set(self, W = None, x = None, func = None, method = None, args = None, norm = True):
        
        # set normalisation parameter
        self.norm = norm

        # set attributes
        if W is not None:
            if hasattr(W, "__len__"):       # if weights given is a vector
                self.W = W.copy()   
            elif isinstance(W, float):      # if weight given is a single float scalar
                self.W = W
            elif isinstance(W, int):        # if weight given is a single int scalar
                self.W = float(W)
            else:
                log("[W] weights instance attribute must be a float or array-like", stype = "warn")
        

        #must be an array if used to evaluate functions
        if x is not None:
            if hasattr(x, "__len__"):
                self.x = x.copy()
            else:
                log("[x] weights instance attribute must be array-like", stype = "warn")

        # check if string, then eval it 
        if func is not None:
            # if func is string
            if isinstance(func, str):
                self.func = eval(func)
        
            # if func is callable, directly set 
            elif callable(func):
                self.func = func
        
            else:
                log("function must either be a string to evaluate, or a callable function.", stype = "err")

        # must choose a method that is supported
        if method is not None:
            if method not in ["interp", "func", "None"]:
                log(f"Invalid Method for Weighting, using {self.method}", stype = "warn")
            else:
                self.method = method    

        # set arguments of callable function as attributes. NOTE: if not all arguments are 
        # specified, then warn the user
        if args is not None:
            if not isinstance(args, dict):
                log("args must be in dict format", stype = "err")
            if self.func is not None:
                func_keys = inspect.getargspec(self.func)[0][1:]
                if_flag = False
                missing_args = []
                new_args = {}
                for key in func_keys:
                    if key in args.keys():
                        new_args[key] = args[key]
                    else:
                        if_flag = True
                        missing_args += key
                
                if if_flag: 
                    log(f"Not all args have been specified for this function {missing_args}", stype = "warn")

                self.args = deepcopy(args)

            else:
                log("weight func not given, can't determine args", stype = "warn")

        return

    

    def get_weights(self, x = None, method = None):
        """
        Get weights, this is a universal function 

        Parameters
        ----------
        x : ndarray or array-like, optional
            if specified, will interp or evaluate func using this array rather then self.x, by default None
        method : str, optional
            method used for retrieving weights, if not spefified, uses self.method, by default None

        Returns
        -------
        W : ndarray or array-like
            Weights
        """        

        if method is not None:
            self.set(method = method)
        
        # get weights from .W attribute
        if self.method == "None":
            outw = self._get_weights_from_W()

        # interp weights based on 
        if self.method == "interp":
            outw = self._get_weights_from_interp(x)

        if self.method == "func":
            outw = self._get_weights_from_func(x)

        # normalise
        if self.norm and (outw is not None) and hasattr(outw, "__len__"):
            Wsum = np.mean(outw)
            if Wsum == 0.0:
                log("Weights sum to zero, cannot normalize!", stype = "warn")
                return outw
            return outw / Wsum

        else:
            return outw




    def _get_weights_from_W(self):
        """
        Get stored weights

        Returns
        -------
        W : float or array-like
            Weights
        """

        if self.W is not None:
            if isinstance(self.W, float):
                return self.W
            elif hasattr(self.W, "__len__"):
                return self.W.copy()
        else:
            log("No Weights specified, returning None object", stype = "err")
            return None
        


    def _get_weights_from_interp(self, x = None):
        """
        Use a reference array [self.x] and compare it to an input array [x] to interpolate 
        The stored weights.

        Parameters
        ----------
        x : array-like, optional
            x values that will be compared to self.x, by default None

        """

        if self.W is None:
            log("No weights specified, returning None object", stype = "err")
            return None

        if self.x is None:
            log("No x array specified for interpolation, for interpolation a weights array [W] with ", stype = "err")
            log("array [x] of same size must be defined in the weights class before interpolation is performed.", stype = "err")
            return None

        # check x
        if x is None:
            log("No x array given for interpolation, returning the full array of weights", stype = "warn")
            return self.W

        # check if bounds of x is larger than [x] saved in weights instance
        if (x[0] < self.x[0]) or (x[-1] > self.x[-1]):
            log("Bounds of specified array x larger than array [x] of weights instance, for any x values outside", stype = "warn")
            log("the known weights range they will be set to the bounded values of the weights array.", stype = "warn")

        # perform interpolation
        return np.interp(x, self.x, self.W)


    def _get_weights_from_func(self, x = None):
        """
        Evaluate function to retrieve weights

        Parameters
        ----------
        x : x values to evaluate function with, optional
            _description_, by default None, if None, will use what is stored i.e. self.x

        """
        
        if not self._check_func_and_args():
            return None

        if x is None:
            
            if self.x is None:
                log("Must specify [x] array", stype = "err")
                return None
            else:
                log("Using stored [x] array of weights instance", stype = "warn")
                return self.func(self.x, **self.args)

        else:
            return self.func(x, **self.args)



    def _check_func_and_args(self):
        """
        Check if function is callable and args are sufficient

        """

        if self.func is None:
            log("Must specify a function", stype = "err")
            return 0

        if self.args is None:
            log("Must specify args for func", stype = "err")
            return 0
        
        if (not callable(self.func)) or (not isinstance(self.args, dict)):
            log("func must be callable and args a dictionary of args to evaluate the function", stype = "err")
            return 0
        
        # check if all args are there
        for key in inspect.getargspec(self.func)[0][1:]:
            if key not in self.args.keys():
                log(f"argument [{key}] not specified, cannot evaulate function", stype = "err")
                return 0

        # if all pass return 1
        return 1




    def __str__(self):
        pstr = ""
        if self.W is not None:
            if hasattr(self.W, "__len__"):
                Ws = f"[{self.W.shape}]"
            elif isinstance(self.W, float):
                Ws = self.W
        else:
            Ws = None
        pstr += "[W]".ljust(10) + f"= {Ws}\n"

        if self.x is not None:
            pstr += "[x]".ljust(10) + f"= {self.x.shape}\n"
        else:
            pstr += "[x]".ljust(10) + "= None\n"

        pstr += "[func]".ljust(10) + f"= {self.func}\n"
        pstr += "[method]".ljust(10) + f"= {self.method}\n"

        if self.args is not None:
            pstr += "[args]".ljust(10) + "=\n"
            for key in self.args.keys():
                pstr += f"[{key}]".ljust(20).rjust(16) + f"{self.args[key]}\n"
        else:
            pstr += "[args]".ljust(10) + "= None\n"

        return pstr























class FRB_params:
    """
    FRB parameter structure 
    
    Attributes
    ----------
    name: str
        name of FRB
    RA: str            
        Right ascension
    DEC: str            
        Declination
    MJD: float
        Modified julian date [days]
    DM: float             
        Dispersion Measure [pc/cm^3]
    bw: float            
        Bandwidth [MHz]
    cfreq: float          
        Central Frequency [MHz]
    t_lim: List          
        Time bounds [ms]
    f_lim: List          
        Frequency bounds [MHz]
    t_ref: float
        Reference point of time-series (0-point)
    dt: float             
        delta time [ms]
    df: float            
        delta frequency [MHz]
    nchan: int         
        Number of channels
    nsamp: int          
        Number of samples
    UP: bool
        Upper bandwidth
    RM: float
        Rotation Measure [Rad/m^2]
    f0: float
        Reference frequency [MHz]
    pa0: float
        Positon angle at f0
    tW: np.ndarray
        time weights
    fW: np.ndarray
        frequency weights



    Parameters
    ----------
    name: str
        name of FRB
    RA: str            
        Right ascension
    DEC: str            
        Declination
    DM: float             
        Dispersion Measure [pc/cm^3]
    MJD: float
        Modified julian date [days]
    bw: float            
        Bandwidth [MHz]
    cfreq: float          
        Central Frequency [MHz]
    t_lim: List          
        Time bounds [ms]
    f_lim: List          
        Frequency bounds [MHz]
    t_ref: float
        Reference point of time-series (0-point)
    dt: float             
        delta time [ms]
    df: float            
        delta frequency [MHz]
    nchan: int         
        Number of channels
    nsamp: int          
        Number of samples
    RM: float
        Rotation Measure [Rad/m^2]
    f0: float
        Reference frequency [MHz]
    pa0: float
        Positon angle at f0
    tW: np.ndarray
        time weights
    fW: np.ndarray
        frequency weights
    
    




    
    """

    def __init__(self, name: str = _G.p['name'],    RA: str = _G.p['RA'],    DEC: str = _G.p['DEC'], 
                       MJD: float = _G.p['MJD'],    DM: float = _G.p['DM'],      bw: int = _G.p['bw'],    
                       cfreq: float = _G.p['cfreq'], 
                       t_lim_base  = _G.p['t_lim_base'],      f_lim_base = _G.p['f_lim_base'],   RM: float = _G.p['RM'],
                       f0: float = _G.p['f0'],      pa0: float = _G.p['pa0'],
                       dt: float = _G.p['dt'],      df: float = _G.p['df'], t_ref: float = _G.p['t_ref'],
                       EMPTY = False):

        # parameters
        self.name   = name              # name of FRB
        self.RA     = RA                # Right Acension 
        self.DEC    = DEC               # Declination
        self.MJD    = MJD               # MJD date 
        self.DM     = DM                # dispersion measure
        self.bw     = bw                # bandwidth
        self.cfreq  = cfreq             # central frequency
        self.RM     = RM                # rotation measure
        self.f0     = f0                # reference frequency
        self.pa0    = pa0               # reference PA
        self.fW     = weights()         # frequency weightings
        self.tW     = weights()         # time weightings

        # define base parameters, these will be used when changing the crop parameters, 
        # all other parameters will be updated from these 

        # crop parameters
        self.t_lim_base  = t_lim_base                                       # time range
        self.f_lim_base  = [cfreq - 0.5*bw, cfreq + 0.5*bw]             # frequency range
        self.t_ref  = t_ref

        # calculate resolutions
        self.dt     = dt                # delta time in ms
        self.df     = df                # delta frequency in MHz

        # dimensions
        self.nchan = None               # number of channels
        self.nsamp = None               # number of time samples

        self.pcol = 'lgreen'

        if EMPTY:
            self.empty_par()


    @property
    def t_lim(self):
        """
        t_lim_ref: t_lim translated by t_ref to the zero point.
        """
        return [self.t_lim_base[0] - self.t_ref, self.t_lim_base[1] - self.t_ref]

    @property
    def f_lim(self):
        return self.f_lim_base.copy()




    def update_from_crop(self, t_crop: list = [0.0, 1.0], f_crop: list = [0.0, 1.0],
                            tN: int = 1, fN: int = 1):
        """
        Update Parameters based on time and frequency crops + averaging

        Parameters
        ----------
        t_crop : list, optional
            time crop, by default [0.0, 1.0]
        f_crop : list, optional
            frequency crop, by default [0.0, 1.0]
        tN : int, optional
            Factor for time averaging, by default 1
        fN : int, optional
            Factor for frequency averaging, by default 1
        """  

        # get starting sample, and number of samples
        t_sampoff = int(self.nsamp * t_crop[0])
        t_nsamp = int(self.nsamp * t_crop[1]) - t_sampoff
        t_nsamp = int(t_nsamp / tN)

        self.t_lim_base[0] = self.t_lim_base[0] + t_sampoff * self.dt
        self.t_lim_base[1] = self.t_lim_base[0] + t_nsamp * self.dt * tN
        self.dt *= tN
        self.nsamp = t_nsamp
        

        # get starting chan and number of chans
        f_sampoff = int(self.nchan * f_crop[0])
        f_nchan = int(self.nchan * f_crop[1]) - f_sampoff
        f_nchan = int(f_nchan / fN)

        self.f_lim_base[1] = self.f_lim_base[1] - f_sampoff * self.df
        self.f_lim_base[0] = self.f_lim_base[1] - f_nchan * self.df * fN      
        self.df *= fN
        self.nchan = f_nchan
        self.bw = self.f_lim_base[1] - self.f_lim[0]
        self.cfreq = self.f_lim_base[0] + self.bw / 2


    
        
    
    # function to save parameters to parameter file
    def phase2lim(self, t_crop: list = None, f_crop: list = None, snap = False):
        """
        Provide time and frequency phases, based on parameters convert
        to ms and MHz crops

        Parameters
        ----------
        t_crop : list, optional
            Time crop, in phase, by default None
        f_crop : list, optional
            Freq crop, in phase, by default None
        snap : bool, optional
            if true, snap to nearest multiple of sample resolution, false by default

        Returns
        -------
        t_lim: List
            Time limits in [ms]
        f_lim: List
            Freq limits in [MHz]
        """        

        # init
        t_lim, f_lim = None, None
        
        # convert phase to limit based on parameters
        if t_crop is not None:
            # time
            t_lim = [0.0, 0.0]
            lim_width = self.t_lim_base[1] - self.t_lim_base[0]
            t_lim[0] = t_crop[0]*(lim_width)
            t_lim[1] = t_crop[1]*(lim_width)
            if snap:
                t_lim[0] = round(t_lim[0] / self.dt) * self.dt
                t_lim[1] = round(t_lim[1] / self.dt) * self.dt
            t_lim[0] += self.t_lim[0]
            t_lim[1] += self.t_lim[0]

        if f_crop is not None:
            # f_lim in ascending freq, convert upperside band crop
            f_crop_flip = [1.0 - f_crop[1], 1.0 - f_crop[0]]

            # frequency
            f_lim = [0.0, 0.0]
            lim_width = self.f_lim_base[1] - self.f_lim_base[0]
            f_lim[0] = f_crop_flip[0]*(lim_width)
            f_lim[1] = f_crop_flip[1]*(lim_width)
            if snap:
                f_lim[0] = round(f_lim[0] / self.df) * self.df
                f_lim[1] = round(f_lim[1] / self.df) * self.df
            f_lim[0] += self.f_lim[0]
            f_lim[1] += self.f_lim[0]

        return t_lim, f_lim

    


    # function to get crop from limits
    def lim2phase(self, t_lim: list = None, f_lim: list = None, snap = False):
        """
        Provide time [ms] and Freq [MHz] limits and using params, convert 
        to time and freq phase.

        Parameters
        ----------
        t_lim: List
            Time limits in [ms]
        f_lim: List
            Freq limits in [MHz]
        snap : bool, optional
            if true, snap to nearest multiple of sample resolution, false by default

        Returns
        -------
        t_crop : list, optional
            Time crop, in phase, by default None
        f_crop : list, optional
            Freq crop, in phase, by default None
        """        
        
        # init
        t_phase, f_phase = None, None

        # convert physical points to phase
        if t_lim is not None:
            # time
            t_phase = [0.0, 0.0]
            lim_width = self.t_lim[1] - self.t_lim[0]
            t_phase[0] = (t_lim[0] - self.t_lim[0])
            t_phase[1] = (t_lim[1] - self.t_lim[0])
            if snap:
                t_phase[0] = round(t_phase[0] / self.dt) * self.dt
                t_phase[1] = round(t_phase[1] / self.dt) * self.dt
            t_phase[0] /= lim_width
            t_phase[1] /= lim_width

        if f_lim is not None:
            # frequency
            f_phase = [0.0, 0.0]
            lim_width = self.f_lim[1] - self.f_lim[0]
            f_phase[0] = (f_lim[0] - self.f_lim[0])
            f_phase[1] = (f_lim[1] - self.f_lim[0])
            if snap:
                f_phase[0] = round(f_phase[0] / self.df) * self.df
                f_phase[1] = round(f_phase[1] / self.df) * self.df
            f_phase[0] /= lim_width
            f_phase[1] /= lim_width

            f_phase = [1.0 - f_phase[1], 1.0 - f_phase[0]]

        return t_phase, f_phase


    
    def mkpar_from_params(self, frb_params: dict = None):
        """
        Make new copy of params instance

        Parameters
        ----------
        frb_params : dict, optional
            keyword parameters, by default None

        Returns
        -------
        params : FRB_params
            New instance of FRB params
        """        
        # make copy of frb params
        frb_params = dict_init(frb_params)

        new_par = self.copy()
        crop_par = {}

        # loop over params
        for key in frb_params:
            if key in _G.p.keys():
                setattr(new_par, key, frb_params[key])
            if key in _G.crop_params.keys():
                crop_par[key] = frb_params[key]
         
        # create new par instance and return 
        return new_par.build_from_crop(**crop_par)


    
    def par2dict(self):
        """
        Return parameters of class as dictionary

        """
        new_params = {}
        for key in _G.p.keys():
            new_params[key] = getattr(self, key)
        new_params["t_lim"] = self.t_lim.copy()
        new_params["f_lim"] = self.f_lim.copy()

        return deepcopy(new_params)



    # print function
    def __str__(self):

        param_str = "===== params =====\n"

        for _,key in enumerate(_G.p.keys()):
            param_str += f"{key}:       {getattr(self,key)}\n"
        
        param_str += f"t_lim:           {self.t_lim}\n"
        param_str += f"f_lim:           {self.f_lim}\n"

        return param_str
    

    # copy function
    def copy(self):
        """
        Return copy of parameter class
        """        
        return deepcopy(self)




    def set_par(self, **kwargs):
        """
        Set attributes of par class

        Parameters
        ----------
        **kwargs : Dict
            Keyword parameters
        
        """
        # get all relevant pars

        for key in kwargs.keys():
            if key in _G.p.keys():
                setattr(self, key, kwargs[key])
        

        self.f_lim_base  = [self.cfreq - 0.5*self.bw, self.cfreq + 0.5*self.bw] # frequency range


    def set_weights(self, xtype = "t", **kwargs):
        """
        Set properties of weights instances

        Parameters
        ----------
        xtype: str
            Type of weights \n
            "t" - Time weights \n
            "f" - Freq weights

        **kwargs: Dict
            Keyword arguments for weights instance
        """

        if xtype == "t": # time weights
            self.tW.set(**kwargs)

        elif xtype == "f": # freq weights
            self.fW.set(**kwargs)
        
        else:
            log("xtype must be either 't' for time, or 'f' for freq", stype = "err")
        
        return 

        
    def get_freqs(self):
        """
        Get frequencies
        """
        return np.linspace(self.cfreq + self.bw/2 - self.df/2, self.cfreq - self.bw/2 + self.df/2,
                            self.nchan)

    def get_times(self):
        """
        Get time bins
        """
        return np.linspace(self.t_lim[0] + 0.5 * self.dt, self.t_lim[1] - 0.5 * self.dt, self.nsamp)


    def empty_par(self):
        """
        Set all parameters to None:
        """

        for key in _G.p.keys():
            setattr(self, key, None)

        
    def default_par(self):
        """
        Set all parameters to default
        """

        for key in _G.p.keys():
            setattr(self, key, _G.p[key])

















##===================================##
## FRB METAPARAMETER CLASS/CONTAINER ##
##===================================##

class FRB_metaparams:
    """
    Class for FRB meta-params

    Attributes
    ----------
    t_crop : List
        Time crop
    f_crop : List
        Frequency crop
    tN : int
        Factor for time averaging
    fN : int
        Factor for frequency averaging
    zapchan: str
        string used for zapping channels, in format -> "850, 860, 870:900" \n
        each element seperated by a ',' is a seperate channel. If ':' is used, user can specify a range of values \n
        i.e. 870:900 -> from channel 870 to 900 inclusive of both.
    norm: str
        Type of normalisation \n
        [max] - normalise using maximum \n
        [absmax] - normalise using absolute maximum \n
        [None] - Skip normalisation



    Parameters
    ----------
    t_crop : List
        Time crop
    f_crop : List
        Frequency crop
    tN : int
        Factor for time averaging
    fN : int
        Factor for frequency averaging
    norm: str
        Type of normalisation \n
        [max] - normalise using maximum \n
        [absmax] - normalise using absolute maximum \n
        [None] - Skip normalisation
    """    

    def __init__(self, t_crop = None, f_crop = None, terr_crop = None,
                 tN = _G.mp['tN'], fN = _G.mp['fN'], norm = _G.mp['norm'],
                 zapchan = _G.mp['zapchan'], EMPTY = False):


        # set crop parameters
        self.t_crop = t_crop
        self.f_crop = f_crop
        self.terr_crop = terr_crop
        self.zapchan = zapchan

        if self.t_crop is None:
            self.t_crop = _G.mp['t_crop']

        if self.f_crop is None:
            self.f_crop = _G.mp['f_crop']

        # set averaging parameters
        self.tN = tN
        self.fN = fN

        self.norm = norm

        if EMPTY:
            self.empty_metapar()

    

    def set_metapar(self, **kwargs):
        """
        Set meta-parameters
        """

        for key in kwargs.keys():
            if key in _G.mp.keys():
                setattr(self, key, kwargs[key])

        


    def metapar2dict(self):
        """
        Return Dictionary of meta-parameters
        """
        
        metapar = {}
        for key in _G.mp.keys():
            metapar[key] = getattr(self, key)
        
        return deepcopy(metapar)


    def copy(self):
        """
        Return Copy of Meta-params instance
        """
        return deepcopy(self)



    def empty_metapar(self):
        """
        Set all meta parameters to None
        """

        for key in _G.mp.keys():
            setattr(self, key, None)
        



