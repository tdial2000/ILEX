import shutil, argparse, os
from ilex.globals import c  # import to get ilex_path

def get_args():
    """
    Get arguments
    
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("config", help = "full filepath of new FRB config file")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # copy default yaml file to new filepath
    shutil.copy(os.path.join(os.environ['ILEX_PATH'], 'files/default.yaml'), args.config)

