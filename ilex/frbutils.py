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

from .utils import load_param_file, save_param_file, check_ruamel_output
from .globals import _G
from ruamel.yaml import comments
from ruamel.yaml.scalarfloat import ScalarFloat
import numpy as np
from ruamel.yaml import YAML
import os



def _load_ruamel_default_constructors():
    yaml = YAML()
    def_file = os.path.join(os.environ['ILEX_PATH'], 
                                        "files/frb_ruamel_yaml_defaults.yaml")
    with open(def_file) as file:
        return yaml.load(file)



def save_params(frb, file):
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

    yaml = YAML()

    initpars, yaml_obj = load_param_file(filename, True, False)
    ruamel_yaml_defaults = _load_ruamel_default_constructors()
    defscalarfloat = ruamel_yaml_defaults['scalarfloat']
    print(defscalarfloat._prec)
    # print(initpars['par']['pa0'].v, type(initpars['par']['pa0']))
    # print(type(ruamel_yaml_defaults['scalarfloat'].v))


    def set_ruamel_item(ruamel_obj, item):

        if type(item) == dict:
            ruamel_obj = comments.CommentedMap(item)
            return ruamel_obj

        if type(item) == str:
            ruamel_obj = item
            return ruamel_obj

        if type(item) == bool:
            ruamel_obj = item
            return ruamel_obj

        if hasattr(item, '__len__'):
            if len(item) == 1:
                item = float(item[0])
            else:
                item = list(item)

        if (type(item) == float) or (type(item) == np.float64):
            if ruamel_obj is None:
                ruamel_obj = ScalarFloat(item, prec =_G.yaml_def_prec)
                print(ruamel_obj._prec)
                return ruamel_obj

            ruamel_obj = ScalarFloat(item, width=_G.yaml_def_width, prec =_G.yaml_def_prec)

        if type(item) == list:
            if ruamel_obj is None:
                ruamel_obj = comments.CommentedSeq(item)
                return ruamel_obj

            ruamel_obj.clear()
            for _, val in enumerate(item):
                if type(val) == np.float64:
                    ruamel_obj.append(float(val))
                else:
                    ruamel_obj.append(val)
            

        return ruamel_obj



    # Set [pars]
    for _, key in enumerate(_G.p):
        item = getattr(frb.par, key)
        initpars['par'][key] = set_ruamel_item(initpars['par'][key], item)

    # set [metapars]
    for _, key in enumerate(_G.mp):
        item = getattr(frb.metapar, key)
        initpars['metapar'][key] = set_ruamel_item(initpars['metapar'][key], item)

    # set [hyperpars]
    for _, key in enumerate(_G.hp):
        if key not in _G.yaml_ignore:
            item = getattr(frb, key)
            initpars['hyperpar'][key] = set_ruamel_item(initpars['hyperpar'][key], item)

    
    # set RM if applicable
    # if "RM" in frb.fitted_params.keys():
    #     print("Saving fitted RM values")
    #     for parkey, fitkey in zip(["RM", "f0", "pa0"], ["rm", "f0", "pa0"]):
    #         print(parkey, fitkey)
    #         val = frb.fitted_params['RM'][fitkey].val
    #         print(val, type(val))
    #         print(initpars['par'][parkey] is None)
    #         initpars['par'][parkey] = set_ruamel_item(initpars['par'][parkey], val)
    #         print(initpars['par'][parkey], type(initpars['par'][parkey]))
    
    # print(initpars['par']['RM'])
    
    # # set component fitting to time weights if applicable
    # if "tscatt" in frb.fitted_params.keys():
    #     print("Saving component fitting params as time-dependent weights")
    #     initpars['weights']['time']['method'] = "func"

    #     initpars['weights']['time']['args'] = {}
    #     nonargs = ["sigma", "npulse"]
    #     for key in frb.fitted_params['tscatt'].keys():
    #         if key not in nonargs:
    #             initpars['weights']['time']['args'][key] = frb.fitted_params['tscatt'][key]
        
    #     npulse = frb.fitted_params['tscatt']['npulse']


    #     initpars['weights']['time']['func'] = f"make_scatt_pulse_profile_func({npulse})"

    # save params file
    if file is not None:
        filename = file
    save_param_file(initpars, filename, yaml_obj = yaml_obj)


    return






