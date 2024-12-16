##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 31/10/2024 (spooky)
##
##
## 
## 
## FRB Utils library
##===============================================##
##===============================================##

from .utils import (load_param_file, save_param_file, check_ruamel_output, 
                    update_ruamel_CommentedMap, update_ruamel_CommentedSeq)
from .globals import _G
from ruamel.yaml import comments
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import numpy as np
from ruamel.yaml import YAML
import os
from yaml import safe_load as base_yaml_save_load
from yaml import safe_dump as base_yaml_save_dump



def _load_ruamel_default_constructors():
    yaml = YAML()
    def_file = os.path.join(os.environ['ILEX_PATH'], 
                                        "files/frb_ruamel_yaml_defaults.yaml")
    with open(def_file) as file:
        return yaml.load(file)


def save_frb_to_param_file(frb, file):
    """
    Save frb class parameters to yaml file (don't look at it, code sucks :( )

    Parameters
    ----------
    frb : ilex.frb.FRB
        frb class instance
    file : str
        yaml file name
    """

    # get params of loaded yaml file, else get defaults
    filename = frb._yaml_file
    if file is None:
        file = filename

    yaml = YAML()

    initpars, yaml_obj = load_param_file(filename, True, False)

    # [pars]
    for key in _G.p:
        update_ruamel_CommentedMap(initpars['par'], key, getattr(frb.par, key))


    # [metapars]
    for key in _G.mp:
        update_ruamel_CommentedMap(initpars['metapar'], key, getattr(frb.metapar, key))


    # [hyperpars]
    for key in _G.hp:
        update_ruamel_CommentedMap(initpars['hyperpar'], key, getattr(frb, key))


    # Set RM if applicable 
    if "RM" in frb.fitted_params.keys():
        print("Saving fitted RM values")
        for parkey, fitkey in zip(["RM", "f0", "pa0"], ["rm", "f0", "pa0"]):
            val = frb.fitted_params['RM'][fitkey].val
            print(fitkey, type(val))
            update_ruamel_CommentedMap(initpars['par'], parkey, val)

    
    # set time weights if tscatt has been fitted for
    if "tscatt" in frb.fitted_params.keys():
        print("Saving fitted Profile as time weights")
        # make function
        update_ruamel_CommentedMap(initpars['weights']['time'], 'func', 
                f"make_scatt_pulse_profile_func({frb.fitted_params['tscatt']['npulse']:d})")

        tscatt_args = {}
        for key in frb.fitted_params['tscatt'].keys():
            if key == "npulse":
                continue
            tscatt_args[key] = frb.fitted_params['tscatt'][key].val
        
        # set functions metapars
        update_ruamel_CommentedMap(initpars['weights']['time'], 'method', "func")
        update_ruamel_CommentedMap(initpars['weights']['time'], 'norm', True)
        update_ruamel_CommentedMap(initpars['weights']['time'], 'args', tscatt_args)
        

    
    # save params
    with open(file, "wb") as F:
        yaml_obj.dump(initpars, F)


    return 

def _make_new_dynspec_plot_properties_file(dynspec_file):

    with open(dynspec_file, "w") as file:
        pass
    


def _get_dynspec_plot_properties_file():

    dynspec_file = os.path.join(os.environ['ILEX_PATH'], "files/_dynspec_plot_properties.yaml")
    if not os.path.exists:
        _make_new_dynspec_plot_properties_file(dynspec_file)

    return dynspec_file
    



# functions for changing plotting properties
def get_dynspec_plot_properties():

    dynspec_file = _get_dynspec_plot_properties_file()
    with open(dynspec_file, 'r') as file:
        properties = base_yaml_save_load(file)

    return properties


# function to save dynspec_plot properties
def set_dynspec_plot_properties(**kwargs):

    properties = get_dynspec_plot_properties()
    for key in kwargs.keys():
        properties[key] = kwargs[key]

    dynspec_file = _get_dynspec_plot_properties_file()
    with open(dynspec_file, "w") as file:
        base_yaml_save_dump(properties, file)

    


