# imports
import numpy as np
from ilex.frb import FRB
import argparse, sys
from ilex.utils import load_param_file, save_param_file, update_ruamel_CommentedMap


def get_args():

    desc = "Save crop of data with a modifiled ilex config file, only crops in time and takes entire freq crop"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-d", help = "filepath to .yaml config file", type = str, required = True)
    parser.add_argument("--buffer", help = "Amount of padding to put on either side of on-pulse (+ off-pulse) window when making new crop [multiplier +X*width].", type = float, default = 0.2)
    parser.add_argument("-f", help = "filename prefix for data and modified ilex config file", type = str, default = None)

    args = parser.parse_args()

    return args





def save_data(yamlfile, buffer = 0.2, filename = None):
    """
    Save just a crop of the data and a modified ilex config file with updated crop params

    Parameters
    ----------
    buffer: float
        Amount of padding to put on either side of on-pulse (+ off-pulse) window when
        making new crop.

    """

    # initilize data
    frb = FRB(yamlfile)

    stk = []
    for s in "IQUV":
        if frb.ds[s] is not None:
            stk += [f"ds{s}"]
    print(f"Stokes parameters currently loaded in: {stk}")


    # make larger crop by taking into account both on-pulse and off-pulse windows
    t_width = (frb.metapar.t_crop[1] - frb.metapar.t_crop[0])
    t_crop_full = frb.metapar.t_crop.copy()
    f_crop_full = ["min", "max"]    # always take full spectrum
    original_t_crop = frb.metapar.t_crop.copy()

    # update t_crop_full if terr_crop is defined
    if frb.metapar.terr_crop is not None:
        print("Adding off-pulse region to final crop")
        print(f"Previous crop: {t_crop_full}")
        if frb.metapar.terr_crop[0] < t_crop_full[0]:
            t_crop_full[0] = frb.metapar.terr_crop[0]
        if frb.metapar.terr_crop[1] > t_crop_full[1]:
            t_crop_full[1] = frb.metapar.terr_crop[1] 
        print(f"Updated crop: {t_crop_full}")
        original_terr_crop = frb.metapar.terr_crop.copy()
    else:
        original_terr_crop = None

    # add buffer
    t_crop_full[0] -= buffer*t_width
    t_crop_full[1] += buffer*t_width

    # check bounds of crop
    print("Checking bounds of data")
    previous_t_crop_full = t_crop_full.copy()
    if frb.crop_units == "physical":
        if t_crop_full[0] < frb.par.t_lim[0]:
            t_crop_full[0] = frb.par.t_lim[0]
            print("LHS crop edge out-of-bounds, corrected to edge of full data")
        if t_crop_full[1] > frb.par.t_lim[1]:
            t_crop_full[1] = frb.par.t_lim[1]
            print("RHS crop edge out-of-bounds, corrected to edge of full data")
    else:
        print(f"Crop units type [{frb.crop_units}] not supported!")
        sys.exit()

    print(f"Old t_crop: {original_t_crop}")
    if original_t_crop[0] < t_crop_full[0]:
        original_t_crop[0] = t_crop_full[0]
    if original_t_crop[1] > t_crop_full[1]:
        original_t_crop[1] = t_crop_full[1]
    print(f"New t_crop: {original_t_crop}")
    
    if original_terr_crop is not None:
        print(f"Old terr_crop: {original_terr_crop}")
        if original_terr_crop[0] < t_crop_full[0]:
            original_terr_crop[0] = t_crop_full[0]
        if original_terr_crop[1] > t_crop_full[1]:
            original_terr_crop[1] = t_crop_full[1]
        print(f"New terr_crop: {original_terr_crop}")
    
    
    print(f"Previous crop: {previous_t_crop_full}")
    print(f"Updated crop: {t_crop_full}")


    # get data
    # force disable all processing other than cropping
    print("Getting crop...")
    data = frb.get_data(stk, get = True, tN = 1, fN = 1, RM = None, zapchan = "", t_crop = t_crop_full,
                        f_crop = f_crop_full, norm = "None", terr_crop = None)
    
    # now save data
    if filename is None:
        filename = str(frb.par.name) + "_crop"

    for s in stk:
        stk_file = f"{filename}_{s}.npy"
        print(f"saving crop [{s}] to [{stk_file}]")
        np.save(stk_file, data[s])

    print(f"Saving modified yaml config file as " + f"{filename}.yaml")
    defpars, yaml_obj = load_param_file(yamlfile, True, False)

    for s in "IQUV":
        if f"ds{s}" in stk:
            update_ruamel_CommentedMap(defpars['data'], f"ds{s}", f"{filename}_ds{s}.npy")
            # defpars[f"ds{s}"] = f"{filename}_ds{s}.npy"
        else:
            update_ruamel_CommentedMap(defpars['data'], f"ds{s}", "")

    # update base of t_lim
    new_crop_width = t_crop_full[1] - t_crop_full[0]
    update_ruamel_CommentedMap(defpars['par'], 't_lim_base', [0, new_crop_width])
    
    # update time reference point
    new_t_ref = frb.par.t_ref - (t_crop_full[0] - frb.par.t_lim[0])
    update_ruamel_CommentedMap(defpars['par'], 't_ref', new_t_ref)

    # update number of samples
    update_ruamel_CommentedMap(defpars['par'], 'nsamp', data[stk[0]].shape[1])

    # update t crop
    update_ruamel_CommentedMap(defpars['metapar'], 't_crop', original_t_crop)

    # update terr crop
    if original_terr_crop is not None:
        update_ruamel_CommentedMap(defpars['metapar'], 'terr_crop', original_terr_crop)

    with open(f"{filename}.yaml", "wb") as F:
        yaml_obj.dump(defpars, F)

    return




if __name__ == "__main__":

    args = get_args()

    # save data
    save_data(args.d, args.buffer, args.f)

    print("save_data.py Completed!")
