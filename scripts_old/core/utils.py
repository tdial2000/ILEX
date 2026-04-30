# utils for scripts
from ..io import ilexIO
from ..frb import FRB
from copy import deepcopy
import os




class desc:
    pass 



# description of save_config
desc.save_config = """
Save the inputs of a config file

"""





def save_config(iparfile = None, oparfile = None, ofile = None, pars = None, overwrite = False):
    """
    Parameters
    ----------
    iparfile: str
        input ILEX config filepath
    oparfile: str
        output ILEX config filepath
    ofile: str
        output filenames for stokes data, will output in the form <ofile>_ds<stk>.npy
    pars: dict
        attributes of ILEX config file to write
    overwrite: bool
        Overwrite iparfile
    """
    datafilepath = None

    if (iparfile is None) and (iparfile is None):
        return
    
    if iparfile is not None:
        if oparfile is not None:
            if ofile is None:
                datafilepath = os.path.splittext(oparfile)[0]
        
            ilexio = ilexIO(filepath = oparfile, datafilepath = datafilepath,
                            frb = FRB(iparfile), overwrite = overwrite)
            ilexio.save()
    
        return
    
    else:
        if oparfile is not None:
            if 'data' not in pars.keys():
                ValueError("Must specify filenames in [par] dict (par['data'] = {'dsI', 'dsQ', etc:}) if creating a new config file from scratch!")
            
            ilexio = ilexIO(filepath = oparfile, stkfiles = deepcopy(pars['data']))
            ilexio.edit_pars(pars = pars)

            return

    return
            


 
