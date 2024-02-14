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

from .logging import log


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
    norm: str
        Type of normalisation \n
        [max] - normalise using maximum \n
        [absmax] - normalise using absolute maximum \n
        [None] - Skip normalisation



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
    bw: float            
        Bandwidth [MHz]
    cfreq: float          
        Central Frequency [MHz]
    t_lim: List          
        Time bounds [ms]
    f_lim: List          
        Frequency bounds [MHz]
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
    norm: str
        Type of normalisation \n
        [max] - normalise using maximum \n
        [absmax] - normalise using absolute maximum \n
        [None] - Skip normalisation
    
    




    
    """

    def __init__(self, name: str = _G.p['name'],    RA: str = _G.p['RA'],    DEC: str = _G.p['DEC'], 
                       DM: float = _G.p['DM'],      bw: int = _G.p['bw'],    cfreq: float = _G.p['cfreq'], 
                       t_lim  = _G.p['t_lim'],      f_lim = _G.p['f_lim'],   RM: float = _G.p['RM'],
                       f0: float = _G.p['f0'],      pa0: float = _G.p['pa0'],fW = None,                   
                       tW = None,                   norm: str = "max",
                       czap = None,                 EMPTY = False):

        # parameters
        self.name   = name              # name of FRB
        self.RA     = RA                # Right Acension 
        self.DEC    = DEC               # Declination
        self.DM     = DM                # dispersion measure
        self.bw     = bw                # bandwidth
        self.cfreq  = cfreq             # central frequency
        self.RM     = RM                # rotation measure
        self.f0     = f0                # reference frequency
        self.pa0    = pa0               # reference PA
        self.fW     = fW                # frequency weightings
        self.tW     = tW                # time weightings
        self.czap   = czap              # frequencies to zap in string form

        # define base parameters, these will be used when changing the crop parameters, 
        # all other parameters will be updated from these 

        # crop parameters
        self.t_lim  = t_lim                                         # time range
        self.f_lim  = [cfreq - 0.5*bw, cfreq + 0.5*bw]             # frequency range

        # calculate resolutions
        self.dt     = 1e-3              # delta time in ms
        self.df     = 1                 # delta frequency in MHz

        # dimensions
        self.nchan = None               # number of channels
        self.nsamp = None               # number of time samples

        self.UP    = True               # flag, True if data is Upper sideband
        self.pcol = 'lgreen'

        if EMPTY:
            self.empty_par()




    

    # function to update from new crop
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

        # TODO: Maybe find a smarter way to do this
        # reverse f_crop if Upper sideband
        # if self.UP:
        #     f_crop = [1.0 - f_crop[1], 1.0 - f_crop[0]]

        # update params given new crop and averaging
        # update resolutions
        self.dt *= tN
        self.df *= fN

        # update t_lim
        # NOTE: Must correct for int casting that is used for phase slicing and averaging chopping
        lim_width = self.t_lim[1] - self.t_lim[0]
        tlim0 = self.t_lim[0]
        self.t_lim[0] =  math.floor(t_crop[0] * self.nsamp) * lim_width / self.nsamp + tlim0
        self.t_lim[1] =  math.floor(t_crop[1] * self.nsamp) * lim_width / self.nsamp + tlim0

        # update f_lim 
        lim_width = self.f_lim[1] - self.f_lim[0]
        flim0 = self.f_lim[0]
        self.f_lim[0] =  math.floor(f_crop[0] * self.nchan) * lim_width / self.nchan + flim0
        self.f_lim[1] =  math.floor(f_crop[1] * self.nchan) * lim_width / self.nchan + flim0

        # update bw and cfreq
        self.bw = self.f_lim[1] - self.f_lim[0]
        self.cfreq = self.f_lim[0] + 0.5*self.bw

        # update nchan and nsamp
        self.nchan = int((math.floor(f_crop[1] * self.nchan) - math.floor(f_crop[0] * self.nchan))/fN)
        self.nsamp = int((math.floor(t_crop[1] * self.nsamp) - math.floor(t_crop[0] * self.nsamp))/tN)



    # function to load paramters from parameter file
    def load_params(self, filename: str = None):
        """
        Load params from file - To implement??

        Parameters
        ----------
        filename : str, optional
            param file name, by default None
        """        

        if filename is None:
            log("Parameter file must be valid", lpf_col = self.pcol)
            return

        # load into object
        with open(filename, 'r') as file:
            params = yaml.safe_load(file)

        for _,key in enumerate(params.keys()):
            setattr(self,key,params[key])

        log(f"Parameters loaded from [{filename}]", lpf_col = self.pcol)

        return 


    
    # function to get limits from crop
    def save_params(self, filename: str = None):
        """
        Save current parameters to file

        Parameters
        ----------
        filename : str, optional
            filename to save params to, by default None
        """        
        
        if filename is None:
            log("parameter file must be valid", lpf_col = self.pcol)
            return

        # build yaml .object
        params = {}
        for _,key in enumerate(_G.p.keys()):
            params[key] = getattr(self, key)
        
        # save object to .yaml file
        with open(filename, 'w') as file:
            yaml.safe_dump(params, file)

        log(f"Parameters saved to [{filename}]", lpf_col = self.pcol)

        return
        
    
    # function to save parameters to parameter file
    def phase2lim(self, t_crop: list = None, f_crop: list = None):
        """
        Provide time and frequency phases, based on parameters convert
        to ms and MHz crops

        Parameters
        ----------
        t_crop : list, optional
            Time crop, in phase, by default None
        f_crop : list, optional
            Freq crop, in phase, by default None

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
            lim_width = self.t_lim[1] - self.t_lim[0]
            t_lim[0] = t_crop[0]*(lim_width) + self.t_lim[0]
            t_lim[1] = t_crop[1]*(lim_width) + self.t_lim[0]

        if f_crop is not None:
            # frequency
            f_lim = [0.0, 0.0]
            lim_width = self.f_lim[1] - self.f_lim[0]
            f_lim[0] = f_crop[0]*(lim_width) + self.f_lim[0]
            f_lim[1] = f_crop[1]*(lim_width) + self.f_lim[0]

        return t_lim, f_lim

    


    # function to get crop from limits
    def lim2phase(self, t_lim: list = None, f_lim: list = None):
        """
        Provide time [ms] and Freq [MHz] limits and using params, convert 
        to time and freq phase.

        Parameters
        ----------
        t_lim: List
            Time limits in [ms]
        f_lim: List
            Freq limits in [MHz]

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
            t_phase[0] = (t_lim[0] - self.t_lim[0]) / lim_width
            t_phase[1] = (t_lim[1] - self.t_lim[0]) / lim_width

        if f_lim is not None:
            # frequency
            f_phase = [0.0, 0.0]
            lim_width = self.f_lim[1] - self.f_lim[0]
            f_phase[0] = (f_lim[0] - self.f_lim[0]) / lim_width
            f_phase[1] = (f_lim[1] - self.f_lim[0]) / lim_width

            if not self.UP:
                f_phase[0], f_phase[1] = 1.0 - f_phase[1], 1.0 - f_phase[0]

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

        return deepcopy(new_params)



    # print function
    def __str__(self):

        param_str = "===== params =====\n"

        for _,key in enumerate(_G.p.keys()):
            param_str += f"{key}:       {getattr(self,key)}\n"

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

        
    def get_freqs(self):
        """
        Get frequencies
        """
        if self.UP:
            return np.linspace(self.cfreq + self.bw/2 - self.df/2, self.cfreq - self.bw/2 + self.df/2,
                            self.nchan)
        else:
            return np.linspace(self.cfreq - self.bw/2 + self.df/2, self.cfreq + self.bw/2 - self.df/2,
                            self.nchan)


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
    norm : str



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
    norm : str
    """    

    def __init__(self, t_crop = None, f_crop = None, terr_crop = None,
                 tN = _G.mp['tN'], fN = _G.mp['fN'], norm = _G.mp['norm'],
                 EMPTY = False):


        # set crop parameters
        self.t_crop = t_crop
        self.f_crop = f_crop
        self.terr_crop = terr_crop

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
        



