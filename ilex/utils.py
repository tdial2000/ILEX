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
from .globals import _G
from copy import deepcopy
from .logging import log


# empty structure
class struct_:
    pass




# TODO: Add header guard to this statement below?? 
# would like to allow for notebook usage

plt.ioff()


def interactive_on():
    log("Turning interactive ON", lpf = False)
    plt.ion()


def interactive_off():
    log("Turning interactive OFF", lpf = False)
    plt.ioff()




##===============================================##
##              load/save functions              ##
##===============================================##


def load_data(datafile: str, mmap = True):

    # option to enable memory mapping
    data = None
    m_mode = None
    if mmap:
        m_mode = "r"

    #load in a .npy file
    data = np.load(datafile,mmap_mode = m_mode)


    return data


def save_data(data, filename: str):

    with open(filename,"wb") as f:
        np.save(f,data)





##===============================================##
##           dict. Utilility functions           ##
##===============================================##

## [ CHECK IF ALL ENTRIES IN DICT HAVE SAME TYPE ] ##
def dict_checktype(dict, t):

    for _,key in enumerate(dict.keys()):
        if type(dict[key]) is not t:
            return 0
        
    return 1


## [ GET LIST OF ITEMS FROM DICT ] ##
def dict_get(dict, keys):

    new_dict = {}

    for key in keys:
        new_dict[key] = dict[key]
    
    return new_dict



def dict_init(*dicts):

    #assign input dictionaries to empty {} object
    out_ = list(dicts)
    for i,dic in enumerate(out_):
        if dic is None:
            out_[i] = {}

    if len(out_) == 1:
        out_ = out_[0]

    return out_


def dict_isall(dictA, dictB):

    # check if entires in dictA equal those in dictB, 
    # if so return 1, else return 0

    keysA = dictA.keys()
    
    for key in keysA:
        if dictA[key] != dictB[key]:
            return 0
        
    return 1


def merge_dicts(*dicts):

    # combine multiple dictionaries together
    dicts = list(dicts)
    odict = {}
    for i, dic in enumerate(dicts):
        odict = {**odict, **dic}
    
    return odict


def dict_null(dic):

    new_dic = deepcopy(dic)

    # set all items in dict to None
    for key in dic.keys():
        new_dic[key] = None

    return new_dic


def get_stk_from_datalist(data_list):

    stk = []
    for data in data_list:
        stk.append(data[-1])
    
    return ''.join(set(stk))














#------------------------------------------------#
# plotting utilities                             #
#------------------------------------------------#

def plotnum2grid(nrows = None, ncols = None, num = None):
    """
    Info:
        Takes the number of axes you want and creates a grid,
        one can constrain either the number of rows or colums to 
        make the grid. If Neither is specified, will create the smallest 
        sqaure grid to fit.
    
    Args:
        nrows (int): number of rows to keep constant
        ncols (int): number of columns to keep constant
        num (int): number of axes to make into grid

    Returns:
        grid_nrows (int): number of rows of new grid
        grid_ncols (int): number of columns of new grid

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










#-----------------------------------------------#
# extra data utility functions                  #
#-----------------------------------------------#

def stitch_components_together(x_list, y_list = None):
    """
    Info:
        Stitch data together, in the case that the x data segments
        are not contiguous, the difference between samples, assuming
        contigous within each segment, will be used to pad segments 
        together with zeros
    Args:
        x_list (list): List of data segments to patch together
        y_list (list): Optional, Also patch y list together, will 
                       follow x patching and stitch data along last axis
    
    Returns:
        x_patch (ndarray): patched x data array
        y_patch (ndarray): Optional, patched y data array

    """

    dx = x_list[0][1] - x_list[0][0]

    x_patch = x_list[0]
    y_patch = None
    if y_list is not None:
        y_patch = y_list[0]

    for i in range(1, len(x_list)):
        # check bounds of x_patch and new segment
        seg_diff = x_patch[-1] - x_list[i][0]

        # just patch together
        if seg_diff <= dx and seg_diff > 0:
            x_patch = np.append(x_patch, x_list[i])
            if y_list is not None:
                y_patch = np.append(y_patch, y_list[i], axis = -1)
            
        elif seg_diff == 0:
            x_patch = np.append(x_patch, x_list[i][1:])
            if y_list is not None:
                y_patch = np.append(y_patch, y_list[i][...,1:], axis = -1)

        else:
            x_interpatch = np.linspace(x_patch[-1] + dx, x_list[i][0] - dx,
                             int((seg_diff - 2*dx)/dx))
            x_patch = np.append(x_patch, x_interpatch)
            x_patch = np.append(x_patch, x_list[i])
            if y_list is not None:
                y_patch = np.append(y_patch, np.zeros((*y_patch.shape[:-1],x_interpatch.size)), axis = -1)
                y_patch = np.append(y_patch, y_list[i], axis = -1)
        
    return x_patch, y_patch



