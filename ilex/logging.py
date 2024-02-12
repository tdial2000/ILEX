#######################################
#                                     #
# Logging function                    #
#                                     #
#                                     #
#######################################

## imports
import inspect
from os import path, getcwd
import os

verbose_file = "_verbose.txt"





## color hashmap ##
TERMINAL_COLORS = {'None':"\033[39m", 'black':"\033[30m", "red":"\033[31m",
                   'green':"\033[32m", 'yellow':"\033[33m", "blue":"\033[34m",
                   'magenta':"\033[35m", 'cyan':"\033[36m", 'lgrey':"\033[37m",
                   'dgrey':"\033[90m", 'lred':"\033[91m", 'lgreen':"\033[92m",
                   'lyellow':"\033[93m", 'lblue':"\033[94m", 'lmagenta':"\033[95m",
                   'lcyan':"\033[96m", 'white':"\033[97m"}



def show_terminal_cols():
    """
    Print out avaliable colors for terminal printing
    """

    print(TERMINAL_COLORS)




## functions ##

def get_filepath(file):
    """
    Get filepath of verbose file
    """

    return path.join(os.environ['ILEX_PATH'], file)


def check_verbosefile():
    """
    Check if verbose file exists, else create new one
    """
    vfile = get_filepath(verbose_file)

    if not vfile:
        with open(vfile, 'w') as f:
            pass
    
    return




def get_verbose():
    """
    Get verbose parameter
    """

    check_verbosefile()
    vfile = get_filepath(verbose_file)

    # opening verbose file
    with open(vfile, 'r') as f:
        r = f.readline()
        if r == "True":
            return True
        elif r == "False":
            return False
    



def set_verbose(verbose):
    """
    Set verbose parameter
    """

    vfile = get_filepath(verbose_file)

    # opening verbose file
    with open(vfile, 'w') as f:
        if verbose:
            f.write("True")
        else:
            f.write("False")
    
















##==========================##
##    LOGGING FUNCTIONS     ##
##==========================##

def log(pstr, stype = "log", lpf = True, lpf_col = 'None', ):
    """
        Print String with a few utility options

        ##==== inputs ====##
        str:            String to print out to console
        stype:          Message type
                        [log] -> print out as is
                        [warn] -> print out as warning
                        [err] -> print out as error
    """

    if not get_verbose():
        return

    # get parent function name
    if lpf:
        fname = inspect.getouterframes(inspect.currentframe())[1][3]
        fname = "[" + fname.replace("_"," ").upper() + "]: "
    else:
        fname = ""
    
    # get type
    log_type = {"log":TERMINAL_COLORS["None"], "warn":TERMINAL_COLORS['lyellow'],
                "err":TERMINAL_COLORS['lred']}
    
    # build string
    pstr = TERMINAL_COLORS[lpf_col] + fname + log_type[stype] + pstr + log_type["log"]

    print(pstr)
    return

 