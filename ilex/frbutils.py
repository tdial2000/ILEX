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

from .data import average
import os
from yaml import safe_load as base_yaml_save_load
from yaml import safe_dump as base_yaml_save_dump
import numpy as np
from .globals import ILEXPATH


def _make_new_dynspec_plot_properties_file(dynspec_file):

    with open(dynspec_file, "w") as file:
        pass
    


def _get_dynspec_plot_properties_file():

    dynspec_file = os.path.join(ILEXPATH, "files/_dynspec_plot_properties.yaml")
    if not os.path.exists:
        _make_new_dynspec_plot_properties_file(dynspec_file)

    return dynspec_file
    



# functions for changing plotting properties
def get_dynspec_plot_properties():

    dynspec_file = _get_dynspec_plot_properties_file()
    with open(dynspec_file, 'r') as file:
        properties = base_yaml_save_load(file)

    if properties is None:
        properties = {}

    return properties


# function to save dynspec_plot properties
def set_dynspec_plot_properties(**kwargs):

    properties = get_dynspec_plot_properties()
    for key in kwargs.keys():
        properties[key] = kwargs[key]

    dynspec_file = _get_dynspec_plot_properties_file()
    with open(dynspec_file, "w") as file:
        base_yaml_save_dump(properties, file)

    

def dynspec_smart_loader(x, t_crop = [0.0, 1.0], f_crop = [0.0, 1.0], tN = 1, fN = 1, log = False):
    """
    Perform segmented cropping/downsampling in time/freq for large datasets to 
    conserve memeory usage and load times.
    
    """
    FILECHUNKSIZE = 200e6
    nchan, nsamp = x.shape

    strexp = f"""t-crop:[{t_crop[0]}, {t_crop[1]}], f-crop:[{f_crop[0]}, {f_crop[1]}], tN: {tN}, fN: {fN}"""

    # time params
    startsamp, endsamp = int(t_crop[0] * nsamp), int(t_crop[1] * nsamp)
    crop_nsamp = endsamp - startsamp
    crop_nsamp = crop_nsamp // tN * tN
    endsamp = startsamp + crop_nsamp

    # freq params
    startchan, endchan = int(f_crop[0] * nchan), int(f_crop[1] * nchan)
    crop_nchan = endchan - startchan
    crop_nchan = crop_nchan // fN * fN
    endchan = startchan + crop_nchan    
    
    # calculate size of segment to load for each iteration

    segsamp = int(FILECHUNKSIZE // crop_nchan // tN * tN)

    # calculate number of segments
    Nseg = crop_nsamp // segsamp

    # check if any nan values
    nanflag = False
    nanstr = ""
    if np.any(np.isnan(x[:, 0])):
        nanflag = True
        nanstr = " NaNs found! averaging non NaNs in Frequency"

    # start loading in data
    xout = np.zeros((crop_nchan // fN, crop_nsamp // tN), dtype = x.dtype)
    xcoarse_segsamp = segsamp // tN
    i = -1
    for i in range(Nseg):
        xseg = x[startchan:endchan, startsamp+i*segsamp:startsamp+(i+1)*segsamp]
        xseg = average(xseg, axis = 1, N = tN)
        xout[:, i*xcoarse_segsamp:(i+1)*xcoarse_segsamp] = average(xseg, axis = 0, N = fN, nan = nanflag)
        if log:
            print(f"Progress: {i/(Nseg+1):.2%} ({i}/{Nseg+1}): " + strexp + nanstr, end = "\r")
    
    # load last seg
    xseg = x[startchan:endchan, startsamp+(i+1)*segsamp:startsamp+crop_nsamp]
    xseg = average(xseg, axis = 1, N = tN)
    xout[:, (i+1)*xcoarse_segsamp:] = average(xseg, axis = 0, N = fN, nan = nanflag)
    if log:
        print(f"Progress: 100.00% ({Nseg+1}/{Nseg+1}): " + strexp + "\n")
    return xout





