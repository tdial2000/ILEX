##===============================================##
##===============================================##
## Author: Tyson Dial                            ##
## Email: tdial@swin.edu.au                      ##
## Last Updated: 09/08/20245                     ##
##                                               ##
##                                               ##
## Input/Output utilities                        ##
##                                               ##           
##===============================================##
##===============================================##
# from .frb import FRB
from .classes import FRB
from .globals import _G
from ruamel.yaml import comments
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import numpy as np
from ruamel.yaml import YAML
import os
from yaml import safe_load as base_yaml_save_load
from yaml import safe_dump as base_yaml_save_dump
from ruamel.yaml.scalarfloat import ScalarFloat as ruamel_float
from .utils import *
from copy import deepcopy


class ilexIO:

    """
    Class for loading and saving ilex .yaml files (config files)

    Attributes
    ----------
    filepath: str
        path to .yaml file
    frb: ilex.frb.FRB
        ilex FRB class
    datafilepath: str
        save stokes data in format <datafilepath>_ds<stk>.npy for stk = "I,Q,U,V"
    stkfiles: dict
        Dictionary of filenames for stokes files, {'dsI': <filepath for dsI>, 'dsQ', etc:}
    
    """
    _ATTR = ["filepath", "datafilepath", "frb", "overwrite",
             "proc", 'stkfiles']

    def __init__(self, filepath = None, frb = None, 
                 overwrite = False, proc = False, datafilepath = None, stkfiles = None):
        """
        constructor 

        if filepath isn't given, a new one will be created
        
        """

        self.filepath = filepath
        self.frb = frb
        self.overwrite = overwrite
        self.datafilepath = datafilepath
        self.proc = proc
        self.stkfiles = stkfiles


    @property
    def frb(self):
        """
        Get FRB object
        """
        
        if self._frb is None:
            return None
        
        if type(self._frb) != FRB:
            TypeError("[.frb] attribute must be of type [ilex.frb.FRB]!")
        
        return self._frb

    
    @frb.setter
    def frb(self, frb = None):

        if frb is None:
            ValueError("Must specify frb argument!")
        
        if type(frb) != FRB:
            TypeError("frb must be of type [ilex.frb.FRB]!")
        
        self._frb = frb



    def set(self, **kwargs):
        """
        Set attributes of ilexIO class
        """

        for key in kwargs.keys():
            if key in self._ATTR:
                setattr(self, key, kwargs[key])
        

    def load_pars(self):
        """
        Get just params from file 
        """
        return load_param_file(self.filepath)


    def edit_pars(self, pars = None):
        """
        Edit attributes of ILEX config file

        """
        yaml = YAML()

        inppars, yamlobj = load_param_file(self.filepath, True, False)

        if pars is not None:
            # edit inppars
            inppars = _edit_pars_dict(inppars, pars)
        
        # dump
        with open(self.filepath, "wb") as F:
            yamlobj.dump(inppars, F)

        return


    

    def load(self):
        """
        Load yaml file, overwrite frb obj

        Returns
        -------
        frb : ilex.frb.FRB
            FRB class instance
        """

        if self.filepath is None:
            ValueError("Must specify a config file to load!")
        
        frb = FRB(self.filepath)
    
        return frb


    def save(self):
        """
        Save frb instance to yaml file
        
        """
        # get yaml file name
        if self.filepath is None:
            self.filepath = f"{self.frb.par.name}.yaml"
            if (self.frb._yaml_file is not None) and (self.overwrite):
                self.filepath = self.frb._yaml_file
        
        print(f"Saving ilex config file as: [{self.filepath}]")

        # get stokes dynspec filenames
        stkfiles = {'dsI': None, 'dsQ': None, 'dsU': None, 'dsV': None}
        if self.datafilepath is None:
            if self.overwrite:
                stkfiles = deepcopy(self.frb._data_files)
            else:
                for s in "IQUV":
                    if self.stkfiles is not None:
                        if self.frb.get_filepaths(s) is not None:
                            if self.stkfiles[f"ds{s}"] is None:
                                ValueError(f"Must specify filepath for [ds{s}] if you want to provide individual stk data filenames")
                            stkfiles[f"ds{s}"] = os.path.join(os.getcwd(), self.stkfiles[f"ds{s}"])
                    else:
                        if self.frb.get_filepaths(s) is not None:
                            stkfiles[f"ds{s}"] = os.path.join(os.getcwd(), f"{self.frb.par.name}_ds{s}.npy")
        else:
            for s in "IQUV":
                if self.frb.get_filepaths(s) is not None:
                    stkfiles[f"ds{s}"] = os.path.join(os.getcwd(), f"{self.datafilepath}_ds{s}.npy")
            

        # save frb instance to yaml file
        print(stkfiles)
        _save_frb_to_param_file(self.frb, yamlfile = self.filepath, 
                                datafiles = stkfiles,
                                proc = self.proc)

        
            
        
        







##===============================================##
##              load/save functions              ##
##===============================================##


def load_data(datafile: str, mmap = True):
    """
    Load data to memory map

    Parameters
    ----------
    datafile : str
        filename or path
    mmap : bool, optional
        is memorymap?, by default True

    Returns
    -------
    data : np.mmap or np.ndarray
        loaded data
    """
    # option to enable memory mapping
    data = None
    m_mode = None
    if mmap:
        m_mode = "r"

    #load in a .npy file
    data = np.load(datafile,mmap_mode = m_mode)


    return data


def save_data(data, filename: str):
    """
    Save data to file

    Parameters
    ----------
    data : np.ndarray
        data to save to file
    filename : str
        filename to save data to
    """
    with open(filename,"wb") as f:
        np.save(f,data)





def _init_pars(p, d, ruamel2py = True):
    """
    p : pars
    d : default pars
    """

    for key in d.keys():
        if key not in p.keys():
            if hasattr(d[key], '__len__'):
                p[key] = deepcopy(d[key])
            else:
                if ruamel2py:
                    p[key] = check_ruamel_input(d[key])
                else:
                    p[key] = d[key]
        
        else:
            # check if dict instance
            # check if ruamel yaml input 
            if ruamel2py:
                p[key] = check_ruamel_input(p[key])

            if isinstance(d[key], dict):
                d[key].fa.set_flow_style()
                _init_pars(p[key], d[key], ruamel2py=ruamel2py)


    return p


def load_param_file(param_file = None, return_yaml_obj = False, ruamel2py = True):
    """
    Load in param file and compare with default params file

    Parameters
    ----------
    param_file : str
        parameter file to load in, if None, will return default yaml file values
    
    Returns
    -------
    params : Dict
        parameters, compared with defaults

    """

    yaml = YAML()

    if param_file is not None:
        # open param file
        with open(param_file) as file:
            pars = yaml.load(file)
    else:
        pars = {}

    # open default param file
    with open(os.path.join(os.environ['ILEX_PATH'], "files/default.yaml")) as deffile:
        def_pars = yaml.load(deffile)
    
    if return_yaml_obj:
        return _init_pars(pars, def_pars, ruamel2py=ruamel2py), yaml
    else:
        return _init_pars(pars, def_pars, ruamel2py=ruamel2py)



def save_param_file(pars, filename, yaml_obj = None):
    """
    save to new parameter file

    Parameters
    ----------
    pars : dict
        dictionary of parameters in ilex.yaml format
    filename : str
        filename of saved yaml file
    
    """

    # class MyDumper(yaml.SafeDumper):
    # # HACK: insert blank lines between top-level objects
    # # inspired by https://stackoverflow.com/a/44284819/3786245
    #     def write_line_break(self, data=None):
    #         super().write_line_break(data)

    #         if len(self.indents) == 1:
    #             super().write_line_break()

    if yaml_obj is None:
        yaml_obj = YAML() 


    # save pars in pars
    with open(filename, 'w') as file:
        yaml_obj.dump(pars, file)



def update_ruamel_CommentedSeq(commented_seq, val):
    """
    Updates the value of the CommentedSeq whilst preserving the flow
    stype and comments

    """

    # copy comments
    if commented_seq is not None:
        comment = commented_seq.ca.comment
    else:
        comment = None

    # create new Commented Sequence initialised to val
    commented_seq = CommentedSeq(val)

    # add comments and preserved flow style
    commented_seq._yaml_add_comment(comment)
    commented_seq.fa.set_flow_style()
    
    return commented_seq



def update_ruamel_CommentedMap(commented_map, key, val):
    """
    Updated Commented map based on key, value pair
    """

    if key not in commented_map.keys():

        if val is None:
            commented_map[key] = None

        # add to commented_map
        if isinstance(val, list):
            # create CommentedSeq
            commented_map[key] = CommentedSeq(val)
            commented_map[key].fa.set_flow_style()
        elif isinstance(val, dict):
            # create CommentedMap
            commented_map[key] = CommentedMap(val)
            commented_map[key].fa.set_flow_style()
        elif isinstance(val, float) or isinstance(val, int) or isinstance(val, str) or isinstance(val, bool):
            # add to map
            commented_map[key] = val
        else:
            raise ValueError("Can only add list, dict, float, int, bool or str to CommentedMap yaml.")
    
        return    
        
        
    commented_map.setdefault(key, {})

    if isinstance(val, np.float64):
        val = float(val)

    if isinstance(val, list):
        for i, _ in enumerate(val):
            if isinstance(val[i], np.float64):
                val[i] = float(val[i])
        
        commented_map[key] = update_ruamel_CommentedSeq(commented_map[key], val)
    elif isinstance(val, dict):
        for dict_key in val.keys():
            if isinstance(val[dict_key], np.float64):
                val[dict_key] = float(val[dict_key])
        
        commented_map[key] = val

    else:
        commented_map[key] = val
        

    return
        



def _load_ruamel_default_constructors():
    yaml = YAML()
    def_file = os.path.join(os.environ['ILEX_PATH'], 
                                        "files/frb_ruamel_yaml_defaults.yaml")
    with open(def_file) as file:
        return yaml.load(file)


def _save_frb_to_param_file(frb, yamlfile, datafiles = None, proc = False):
    """
    Save frb class parameters to yaml file (don't look at it, code sucks :( )

    Parameters
    ----------
    frb : ilex.frb.FRB
        frb class instance
    file : str
        yaml file name
    proc : bool
        if True, will process the data before saving to yaml file, will update yaml file accordingly
    """

    # get params of loaded yaml file, else get defaults
    filename = frb._yaml_file
    # if yamlfile is None:
    #     yamlfile = filename
    # if yamlfile is None:
    #     name = frb.par.name
    #     if type(name) != str:
    #         name = "ilex"
    #     yamlfile = f"{name}.yaml"

    yaml = YAML()

    outpars, yaml_obj = load_param_file(filename, True, False)

    inpars = {}
    if proc:
        inpars = _reinitialize_and_save_new_frb_config(frb, datafiles)
        
    else:
        inpars['par'] = frb.par.par2dict()
        inpars['metapar'] = frb.metapar.metapar2dict()
        inpars['hyperpar'] = frb.get_hyperpars()
        if datafiles is not None:
            inpars['data'] = datafiles
        else:
            inpars['data'] = frb._data_files


    # [filepaths]
    for key in inpars['data'].keys():
        update_ruamel_CommentedMap(outpars['data'], key, inpars['data'][key])


    # [pars]
    for key in _G.p:
        update_ruamel_CommentedMap(outpars['par'], key, inpars['par'][key])


    # [metapars]
    for key in _G.mp:
        update_ruamel_CommentedMap(outpars['metapar'], key, inpars['metapar'][key])


    # [hyperpars]
    for key in _G.hp:
        update_ruamel_CommentedMap(outpars['hyperpar'], key, inpars['hyperpar'][key])


    # Set RM if applicable 
    if "RM" in frb.fitted_params.keys():
        print("Saving fitted RM values")
        for parkey, fitkey in zip(["RM", "f0", "pa0"], ["rm", "f0", "pa0"]):
            val = frb.fitted_params['RM'][fitkey].val
            print(fitkey, type(val))
            update_ruamel_CommentedMap(outpars['par'], parkey, val)

    
    # set time weights if tscatt has been fitted for
    if "tscatt" in frb.fitted_params.keys():
        print("Saving fitted Profile as time weights")
        # make function
        if frb.fitted_params['tscatt']['fitmode'] == "abs":
            update_ruamel_CommentedMap(outpars['weights']['time'], 'func', 
                f"make_scatt_pulse_profile_func({frb.fitted_params['tscatt']['npulse']:d})")
        else:
            update_ruamel_CommentedMap(outpars['weights']['time'], 'func', 
                f"make_scatt_pulse_profile_relative_func({frb.fitted_params['tscatt']['npulse']:d})")

        tscatt_args = {}
        for key in frb.fitted_params['tscatt'].keys():
            if key in ["npulse", "sigma", "fitmode"]:
                continue
            tscatt_args[key] = frb.fitted_params['tscatt'][key].val
        
        # set functions metapars
        update_ruamel_CommentedMap(outpars['weights']['time'], 'method', "func")
        update_ruamel_CommentedMap(outpars['weights']['time'], 'norm', True)
        update_ruamel_CommentedMap(outpars['weights']['time'], 'args', tscatt_args)
        

    
    # save params
    with open(yamlfile, "wb") as F:
        yaml_obj.dump(outpars, F)


    return 




def _reinitialize_and_save_new_frb_config(frb, datafiles):
    """
    Reinitialize frb params of a given FRB instance. This function takes the preprocessed data
    and treats it as if it is the orignal dynamic spectrum, thus all intrinsic parameters such as
    time resolution, freq resolution, MJD will be changed whilst pre-processing parameters are reset. i.e.
    dt = dt * tN
    tN = 1
    MJD = MJD + t_crop[0] / 86400000
    t_crop = ['min'. 'max']
    """

    frb._load_new_params()

    t_crop, f_crop = frb.par.phase2lim(t_crop = frb.this_metapar.t_crop,
                                       f_crop = frb.this_metapar.f_crop)
    terr_crop = None
    if frb.this_metapar.terr_crop is not None:
        terr_crop, _ = frb.par.phase2lim(t_crop = frb.this_metapar.terr_crop)

    # update time crops in case terr_crop is specified
    full_tcrop = [*t_crop]
    if terr_crop is not None:
        full_tcrop = [min(t_crop[0], terr_crop[0]),
                      max(t_crop[1], terr_crop[1])]
    
    frb._load_new_params(t_crop = full_tcrop)

    # get params
    outpars = {}
    outpars['par'] = frb.this_par.par2dict()
    outpars['metapar'] = frb.metapar.metapar2dict()
    outpars['hyperpar'] = frb.get_hyperpars()
    
    width = full_tcrop[1] - full_tcrop[0]
    zero = full_tcrop[0] + outpars['par']['t_ref']

    # update!!
    # t_ref
    if outpars['par']['t_ref'] > zero:
        outpars['par']['t_ref'] -= zero
        deltref = 0.0
    else:
        deltref = zero - outpars['par']['t_ref']
        outpars['par']['t_ref'] = 0.0
    
    # t_lim_base
    outpars['par']['t_lim_base'] = [0.0, width]

    # MJD
    outpars['par']['MJD'] += zero / 86400000

    # t_crop
    outpars['metapar']['t_crop'] = [t_crop[0] - deltref,
                                    t_crop[1] - deltref]
    # terr_crop
    if frb.metapar.terr_crop is not None:
        outpars['metapar']['terr_crop'] = [terr_crop[0] - deltref,
                                           terr_crop[1] - deltref]
    
    # f_lim_base
    outpars['metapar']['f_lim_base'] = frb.this_par.f_lim





    # reset!!
    # tN
    outpars['metapar']['tN'] = 1
    outpars['metapar']['fN'] = 1
    outpars['metapar']['f_crop'] = ['min', 'max']


    # datafiles and save data
    outpars['data'] = deepcopy(datafiles)

    stk2load = []
    for s in "IQUV":
        if frb.get_filepaths(s) is not None:
            stk2load += [f"ds{s}"]

    frb.get_data(stk2load, t_crop = full_tcrop)

    for s in "IQUV":
        if f"ds{s}" in stk2load:
            print(f"Saving stk [{s}] as: [{datafiles[f'ds{s}']}]...")
            np.save(datafiles[f"ds{s}"], frb._ds[s])


    return outpars





# will need to make a recursive version later that can take any structure!

def _edit_pars_dict(pars1, pars2):
    """
    Parameters

    Only editing datafiles, pars, metapars and hyperpars is possible at the moment!
    ----------
    pars1: dict[ruamel.yaml]
        dictionary of ILEX config attributes created using ruamel.yaml to preserve comments and structure. This function assumes pars1 has the complete list of ILEX config attributes!
    pars2: dict[ruamel.yaml]
        dictionary of ILEX config attributes to overwrite with in 'pars1'

    Returns
    -------
    pars1: Edited pars1
    """
    # datafiles
    if 'data' in pars2.keys():
        for key in pars2['data'].keys():
            update_ruamel_CommentedMap(pars1['data'], key, pars2['data'][key])


    # pars
    if 'par' in pars2.keys():
        for key in pars2['par'].keys():
            update_ruamel_CommentedMap(pars1['par'], key, pars2['par'][key])

    # metapars
    if 'metapar' in pars2.keys():
        for key in pars2['metapar'].keys():
            update_ruamel_CommentedMap(pars1['metapar'], key, pars2['metapar'][key])

    # hyperpars
    if 'hyperpar' in pars2.keys():
        for key in pars2['hyperpar'].keys():
            update_ruamel_CommentedMap(pars1['hyperpar'], key, pars2['hyperpar'][key])


    return pars1





# #-----------------------------------------------#
# # extra data utility functions                  #
# #-----------------------------------------------#


def check_ruamel_input(inp):
    """
    Ruamel yaml is used in some cases, this will be used to process these inputs and make sure

    Parameters
    ----------
    inp : _class_
        Change ruamel class to python class
    """

    if type(inp) == comments.CommentedMap:
        return dict(inp)
    
    if type(inp) == comments.CommentedSeq:
        return list(inp)

    if type(inp) == ruamel_float:
        return float(inp)
    
    return inp


def check_ruamel_output(out):
    """
    Check if outputs are in right types

    
    """

    if type(out) == float:
        return ruamel_float(out)
    elif type(out) == list:
        return comments.CommentedSeq(out)
    
    return out