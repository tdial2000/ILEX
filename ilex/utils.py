##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 25/09/2023 
##
##
## 
## 
## Utils library
##===============================================##
##===============================================##
import matplotlib.pyplot as plt
import numpy as np
from .globals import _G, ILEXPATH
from copy import deepcopy, copy
from .logging import log
import os
import yaml as base_yaml        # default yaml class


# empty structure
class struct_:
    pass




# TODO: Add header guard to this statement below?? 
# would like to allow for notebook usage

plt.ioff()


# def interactive_on():
#     log("Turning interactive ON", lpf = False)
#     plt.ion()


# def interactive_off():
#     log("Turning interactive OFF", lpf = False)
#     plt.ioff()


##===============================================##
##           dict. Utilility functions           ##
##===============================================##

## [ CHECK IF ALL ENTRIES IN DICT HAVE SAME TYPE ] ##
def dict_checktype(dict, t):
    """
    Check if all entries of Dict are type = t

    Parameters
    ----------
    dict : Dict
        Dictionary
    t : Type()
        type to check against

    Returns
    -------
    _type_
        _description_
    """

    for _,key in enumerate(dict.keys()):
        if type(dict[key]) is not t:
            return 0
        
    return 1


## [ GET LIST OF ITEMS FROM DICT ] ##
def dict_get(dict, keys):
    """
    Get a sub-set of the Dictionary based on keys

    Parameters
    ----------
    dict : Dict
        Original dictionary
    keys : List(str)
        List of items to retrieve

    Returns
    -------
    new_dict : Dict
        Sub-set of Dictionary
    """    
    new_dict = {}

    for key in keys:
        new_dict[key] = dict[key]
    
    return new_dict



def dict_init(*dicts):
    """
    Initilise any number of dictionaries

    Returns
    -------
    *dicts
        A number of Dictionaries
    """    

    #assign input dictionaries to empty {} object
    out_ = list(dicts)
    for i,dic in enumerate(out_):
        if dic is None:
            out_[i] = {}

    if len(out_) == 1:
        out_ = out_[0]

    return out_


def dict_isall(dictA, dictB):
    """
    Check if Entires in Dict A match those in Dict B

    Parameters
    ----------
    dictA : Dict
        Dict A
    dictB : Dict
        Dict B

    Returns
    -------
    bool
        1 if True, 0 if False
    """
    # check if entires in dictA equal those in dictB, 
    # if so return 1, else return 0

    keysA = dictA.keys()
    
    for key in keysA:
        if dictA[key] != dictB[key]:
            return 0
        
    return 1


def merge_dicts(*dicts):
    """
    Combine multiple Dictionaries into 1 dictionary

    Returns
    -------
    *dicts : 
        A number of dictionaries
    """    

    # combine multiple dictionaries together
    dicts = list(dicts)
    odict = {}
    for i, dic in enumerate(dicts):
        odict = {**odict, **dic}
    
    return odict


def dict_null(dic):
    """
    Set all entries in a Dictionary to 'None'

    Parameters
    ----------
    dic : Dict
        Dictionary

    Returns
    -------
    dic : Dict
        Dict with entries set to 'None'
    """    

    new_dic = deepcopy(dic)

    # set all items in dict to None
    for key in dic.keys():
        new_dic[key] = None

    return new_dic

def dict_edit_and_copy(dic, dic2):
    """
    Edit dic entries with select dic2 entries and return a copy of the edited dictionary. 
    If an entry in dic2 exists in dic, that entry will be overwritten. If the entry doesn't exist in 
    dic, the entry is added.

    Be aware that this function will perform a deepcopy of dictiory entries, but only perform a shallow copy of objects!

    Parameters
    ----------
    dic : Dict  
        Dictionary to edit
    dic2 : Dict
        Dictionary to edit entries of dic

    Returns
    -------
    dic : Dict
        Dictionary with altered entries    
    
    """

    new_dic = deepcopy(dic)

    for key in dic2.keys():
        if type(dic2[key]) in [list, dict]:
            new_dic[key] = deepcopy(dic2[key])
        else:
            new_dic[key] = dic2[key]
    
    return new_dic


def get_stk_from_datalist(data_list):
    """
    Get Stokes characters, i.e. I, Q, U and V from list of Stokes 
    products to make, i.e. "dsI", "fQ" etc.

    Parameters
    ----------
    data_list : List(str)
        List of data products to make

    Returns
    -------
    stk : str
        Stokes Characters
    """
    stk = []
    for data in data_list:
        stk.append(data[-1])
    
    stk_s = ''.join(set(stk))
    if "L" in stk_s:
        stk_s = stk_s.replace("L", "")
    if "P" in stk_s:
        stk_s = stk_s.replace("P", "")

    
    return stk_s














#------------------------------------------------#
# plotting utilities                             #
#------------------------------------------------#

def plotnum2grid(nrows = None, ncols = None, num = None):
    """
    Takes the number of axes you want and creates a grid,
    one can constrain either the number of rows or colums to 
    make the grid. If Neither is specified, will create the smallest 
    sqaure grid to fit.
    
    Parameters
    ----------
    nrows : int 
        number of rows to keep constant
    ncols : int
        number of columns to keep constant
    num : int
        number of axes to make into grid

    Returns
    -------
    grid_nrows : int
        number of rows of new grid
    grid_ncols : int 
        number of columns of new grid

    """
    # constraining the number of rows
    if nrows is not None:
        if num <= nrows:
            grid_nrows = num
            grid_ncols = 1

        else:
            grid_nrows = nrows
            grid_ncols = int(num/nrows) + 1
        
    # constraining the number of columns
    elif ncols is not None:
        if num <= ncols:
            grid_ncols = num
            grid_nrows = 1
        
        else:
            grid_ncols = ncols
            grid_nrows = int(num/ncols) + 1

    # conform to nearest square grid
    elif nrows is None and ncols in None:
        n2 = 1 # length of square grid
        while True:
            grid_num = n2**2
            if num <= grid_num:
                grid_ncols = n2
                grid_nrows = n2
                break
        
    else:
        print("Only nrows or ncols can be constrained, or specifiy neither and build aa square grid")
        return (None, ) * 2
    
    return grid_nrows, grid_ncols








def _make_new_plotstyle_file():

    with open(os.path.join(ILEXPATH, "files/_plotstyle.txt"), 'w') as file:
        pass

    return


def load_plotstyle():
    """
    Load in plotstyle file and compare with default plotstyle file
    """

    _default_plotstyle_file = os.path.join(ILEXPATH, "files/_plotstyle.txt")
    if not os.path.isfile(_default_plotstyle_file):
        _make_new_plotstyle_file()

    # read in _plotstyle file
    with open(_default_plotstyle_file, 'r') as file:
        line = file.readline().split(':')
        filepath = line[1].strip()

    if len(filepath) > 0:
        # will use given file
        plotstyle_file = filepath

    else:
        # will use default file
        plotstyle_file = os.path.join(ILEXPATH, "files/default_plot_param_file.yaml")
    

    with open(plotstyle_file) as plf:
        plot_pars = base_yaml.safe_load(plf)

    # check if they match avalaible
    # scatter
    scatter_keys = plot_pars['scatter'].keys()
    for key in scatter_keys:
        if key not in _G.scatter_args:
            del plot_pars['scatter'][key]

    # plot
    plot_keys = plot_pars['plot'].keys()
    for key in plot_keys:
        if key not in _G.plot_args:
            del plot_pars['plot'][key]

    # errorbars
    errorbar_keys = plot_pars['errorbar'].keys()
    for key in errorbar_keys:
        if key not in _G.errorbar_args:
            del plot_pars['errorbar'][key]
    
    return plot_pars


    
def set_plotstyle(filepath = None):
    """
    set plotstyle path, if not given, will be set to default.

    Parameters
    ----------
    filepath : str, optional
        filepath to plotstyle yaml file, by default None
    """

    if filepath is None:
        filepath = ""

    _default_plotstyle_file = os.path.join(ILEXPATH, "files/_plotstyle.txt")
    if not os.path.isfile(_default_plotstyle_file):
        _make_new_plotstyle_file()

    # read in _plotstyle file
    with open(_default_plotstyle_file, 'w') as file:
        line = file.write(f"filepath: {filepath}")

    return




def fix_ds_freq_lims(lims, df):
    """ fix ds frequency limits so they extend over the full dynamic spectrum. By default each frequency channel 
    represents the midpoint of a frequency bin, but when plotting the dynamic spectra using plt.imshow, we need the
    entire bandwidth, so we add half a sample to each end.

    Parameters
    ----------
    lims : list[float]
        frequency limits, [min, max]
    df : float
        frequency resolution
    """

    fixed_lims = lims.copy()
    fixed_lims[0] -= 0.5 * df
    fixed_lims[1] == 0.5 * df

    return fixed_lims


def sort_legend(ax):

    handles, labels = ax.get_legend_handles_labels()

    # sort
    labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))

    return handles, labels