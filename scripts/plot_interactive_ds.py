## imports
from ilex.frb import FRB
import matplotlib.pyplot as plt 
import numpy as np
import argparse
from ilex.data import pslice

# empty class
class _empty:
    pass


def get_args():
    """ get arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument("--parfile", help = "parameter file containning info about FRB", type = str)
    parser.add_argument("-S", help = "Stokes parameter", type = str, default = "S")

    return parser.parse_args()



def plot_ds_int(args):
    """Plot interactive ds with time profile and freq spectra"""
    
    print(f"Plotting stokes {args.S} dynamic spectrum in interactive mode")

    #-------------------------------------#
    # create figure and interactive class #
    #-------------------------------------#

    fig = None
    AX  = None
    dynI = None

    # frb instance
    frb = FRB()
    frb.load_data(yaml_file=args.parfile)

    ## interactive functions
    class zoom_cl:
        """
        Class to implement event handles when zooming in on dynamic spectra
        """
        flag_X = False
        flag_Y = False

        ## update xlims
        def on_zoom_X(zooms,event):
            zooms.flag_X = True
            zooms.update_zoom(event)

        ## update ylims
        def on_zoom_Y(zooms,event):
            zooms.flag_Y = True
            zooms.update_zoom(event)

        ## update profile plots
        def update_zoom(self_zoom,event):
            if not self_zoom.flag_X or not self_zoom.flag_Y: # only update when both x and y lims have changed
                return
            
            self_zoom.flag_X, self_zoom.flag_Y = False, False


            new_X = event.get_xlim()        # get x lims
            new_Y = event.get_ylim()        # get y lims

            t_crop, f_crop = frb.this_par.lim2phase(new_X, new_Y)


            dat = pslice(data[f"ds{args.S}"], *t_crop, axis = 1)                               # get crop of data to create
            dat = pslice(dat, *f_crop, axis = 0)                                # time and freq profiles
            
            AX[1].clear()       # clear time profile axis and plot new crop
            AX[1].plot(np.linspace(new_X[0],new_X[1],dat.shape[1]),np.mean(dat,axis = 0),
                    color = 'k')
            AX[1].set_xlim(new_X)
            AX[1].set_ylabel("Flux Density (arb.)")
            AX[1].set_title(f"Stoke {args.S}")

            AX[2].clear()
            AX[2].plot(np.mean(dat,axis = 1),np.linspace(new_Y[1],new_Y[0],dat.shape[0]),
                    color = 'k')
            AX[2].plot([0.0, 0.0], new_Y[::-1])
            AX[2].set_ylim(new_Y)

    zoom_cb_struct = zoom_cl()

    #create figure
    fig = plt.figure(figsize = (10,10))  
    AX = [] 
    AX.append(fig.add_axes([0.1,0.1,0.7,0.7]))                              # add dynamic spectra axes 
    AX.append(fig.add_axes([0.1,0.8,0.7,0.1]))                              # add time profile axes
    AX.append(fig.add_axes([0.8,0.1,0.1,0.7]))                              # add freq profile axes

    # dynamic spectra
    AX[0].set_ylabel("Freq [MHz]",fontsize = 12)                            # add y label 
    AX[0].set_xlabel("Time [ms]",fontsize = 12)                             # add x label

            
    AX[0].callbacks.connect('xlim_changed',zoom_cb_struct.on_zoom_X)        # x lim event handle
    AX[0].callbacks.connect('ylim_changed',zoom_cb_struct.on_zoom_Y)        # y lim event handle
    
    # time profile plot
    AX[1].get_xaxis().set_visible(False)                                    # turn time series profile axes off

    # frequency profile plot
    AX[2].get_yaxis().set_visible(False)


    # get data
    data = frb.get_data([f"ds{args.S}", f"t{args.S}", f"f{args.S}"], get = True)

    # plot dynamic spectrum
    AX[0].imshow(data[f"ds{args.S}"], aspect = 'auto', extent = [*frb.this_par.t_lim, *frb.this_par.f_lim])


    # return struct to keep data in memory
    return_ = _empty()
    return_.fig = fig
    return_.AX  = AX
    return_.zoom_interactive_struct = zoom_cb_struct

    return return_




if __name__ == "__main__":
    # main block of code

    args = get_args()

    # run interactive plot
    return_ = plot_ds_int(args)

    # plot
    plt.show()