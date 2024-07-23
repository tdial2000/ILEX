import shutil, argparse, os, sys
from ilex.globals import c  # import to get ilex_path
from ilex.utils import load_param_file, save_param_file, dict_get

def get_args():
    """
    Get arguments
    
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("config", help = "full filepath of new FRB config file")

    parser.add_argument("--celebi", help = "Construct yaml config file based on a CELEBI FRB summary txt file", default = None)

    return parser.parse_args()






def make_config_from_celebi(args):
    """
    Make a config file based on an FRB summary txt file currently outputted by the CELEBI pipeline for FRB localisation
    and beamforming. 

    """

    # load in default params file
    defpars, yaml_obj = load_param_file(return_yaml_obj = True)

    # load in a handful of parameters from celebi file
    # read as full string and split using "#", this should split the txt file into sections 
    # we can easily manage
    with open(args.celebi, 'r') as file:
        fullstr = file.read().split('#')
    
    
    for i, section_str in enumerate(fullstr):
        # load in data filepaths
        if "HTR DATA FILEPATHS" in section_str:
            # if this is true, then the next element in this list must have the filepaths
            lines = fullstr[i+1].split('\n')

            print(lines)

            for i, S in enumerate("IQUV"):
                ds_filepath = lines[i+1].split(':')[1].strip()
                print(f"Saving filepath for stokes {S} ds: {ds_filepath}")
                defpars['data'][f'ds{S}'] = ds_filepath

        # load general data like cfreq, DM, FRB name etc. 
        if "GENERAL DATA" in section_str:
            print("\nSAVING PARAMETERS:")
            lines = fullstr[i+1].split('\n')

            yaml_float_pars = ["cfreq", "bw", "DM", "MJD"]

            for line in lines:  # probably not the best way of doing it, but there are different typings
                if "FRB name" in line:
                    frb_name = line.split(':')[1].strip()
                    print(f"name: {frb_name}")
                    defpars['par']['name'] = frb_name

                for i, par in enumerate(["cfreq", "bw", "corr_DM", "crop_MJD"]):
                    if par in line:
                        val = float(line.split(":")[1].split()[0])
                        print(f"{yaml_float_pars[i]}:".ljust(10) + f"{val}")
                        defpars['par'][yaml_float_pars[i]] = val
            
        # load Position pars
        if "POSITION" in section_str:
            print("\nSAVING POSITION:")
            lines = fullstr[i+1].split('\n')

            for line in lines:
                if "RA" in line:
                    RA = line.split()[1]
                    print(f"RA:".ljust(10) + RA)
                    defpars['par']['RA'] = RA
                if "DEC" in line:
                    DEC = line.split()[1]
                    print(f"DEC:".ljust(10) + DEC)
                    defpars['par']['DEC'] = DEC


    # save updated pars to new yaml file
    print(f"Saving new config file to {args.config}")
    save_param_file(defpars, args.config, yaml_obj = yaml_obj)

    


            
        


if __name__ == "__main__":

    args = get_args()

    if args.celebi is not None:

        make_config_from_celebi(args)
        sys.exit()



    # copy default yaml file to new filepath
    shutil.copy(os.path.join(os.environ['ILEX_PATH'], 'files/default.yaml'), args.config)

