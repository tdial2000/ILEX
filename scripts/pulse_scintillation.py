# imports
import matplotlib.pyplot as plt 
import numpy as np 
from ilex.frb import FRB
import argparse

def get_args():

    desc = """
    Model the scintillation over the duration of the pulse. Split the pulse into N different
    time bins and measure the ACF of each, calculating the modulation index as you go. Takes in the arguments
    of the FRB.fit_scintband2() function.
    """

    parser = argparse.ArgumentParser(description = desc, formatter_class = argparse.RawTextHelpFormatter)

    # inputs
    parser.add_argument('--parfile', help = "ILEX config file (.yaml)", type = str, required = True)
    
    # processing arguments
    parser.add_argument('-Nbins', help = "Split the pulse into N time bins", type = int, 
                        default = 1)

    # scintband fitting
    parser.add_argument('--intrinsic_removal', help = "Method for removing intrinsic spectral pulse shape", type = str, default = 'none')
    parser.add_argument('-n', help = "Order of polynomial to fit broad spectrum", type = int, default = None)
    parser.add_argument('--maxsb', help = "Maximum scintillation bandwidth to fit for", type = float, default = 10.0)
    parser.add_argument('--maxlag', help = "Maximum spectral lag to fit scintillation bandwidth", type = float, default = None)

    # additonal arguments
    # parser.add_argument('-v', help = "Verbose mode, makes more diagnostic plots", action = "store_true")
    parser.add_argument('-p', help = "Show plots", action = "store_true")
    parser.add_argument('-r', help = "Redo scintband fitting", action = "store_true")

    # outputs 
    # parser.add_argument('-o', help = "Output directory", type = str, default = None)
    parser.add_argument('-f', help = "Filename prefix", type = str, default = None)

    
    return parser.parse_args()



def fit_scintband(args):
    """
    Fit scintband for each time bin
    """

    # load frb data
    frb = FRB(args.parfile)
    frb.set(show_plots = False, save_plots = True)

    # split into N bins
    binedges = np.linspace(*frb.metapar.t_crop, args.Nbins+1)

    post = {'w': [], 'a': [], 'h': [], 'm': [],
            'werr': [],'aerr': [], 'herr': [], 'merr': [], 't': []}

    # setup priors
    priors = {'w': [0.0, args.maxsb], 'a':[0.0, 1.0], 'h':[0.0, 1.0]}

    print("Fitting scintillation bandwidth...")
    print("-"*40 + "\n")

    for i in range(args.Nbins):
        print(f"Time bin {i}...")
        fit_params = {'outdir':args.f, 'label': f'timebin_{i}'}
        tcrop = [binedges[i], binedges[i+1]]
        p = frb.fit_scintband2(priors = priors, fit_params = fit_params, redo = args.r,
                filename = f"timebin_{i}", n = args.n, intrinsic_removal = args.intrinsic_removal,
                maxlag = args.maxlag, t_crop = tcrop.copy())
        
        for par in ['w','a','h']:
            post[par] += [p.posterior[par].val]
            post[f'{par}err'] += [[p.posterior[par].p, p.posterior[par].m]]
        post['m'] += [p.posterior['m'].val]
        post['merr'] += [p.posterior['m'].p] # pos and neg errors same in this function
        post['t'] += [(tcrop[1] + tcrop[0])/2]
    
    plt.close('all')
    return post, frb




def plot_diagnostics(post, frb):
    """
    Make plots
    """

    print("Plotting results...")

    # make plot of dynspec and time series
    fig, ax = frb.plot_data(['tI', 'dsI'], plot_labels = False)
    ax2 = ax['t0'].twinx()

    ax2.scatter(post['t'], post['m'], marker = 'o', c = 'r', fc = 'k',
                    s = 80, alpha = 0.6)
    ax2.errorbar(post['t'], post['m'], yerr = post['merr'],
                    linestyle = '', capsize = 3.0, ecolor = 'k')
    ax2.set_ylabel("modulation index", color = 'red')
    ax2.tick_params(axis = 'y', labelcolor = 'r')
    ax2.set_ylim([-0.1, 1.1])

    fig.tight_layout()

    fig.savefig(args.f + "_pulse_scintillation.png")


    if args.p:
        plt.show()
    


if __name__ == "__main__":
    # main block of code
    args = get_args()

    # fit scintillation bandwidths
    post, frb = fit_scintband(args)

    # Make pretty plots
    plot_diagnostics(post, frb)

    print("pulse_scintillation.py Completed!")