####################################################
#
#       Python Script with utility functions
#       to enable the use of SHRINE with general
#       ilex/numpy Stokes inputs.
#
#
####################################################
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fftpack import dct, idct
from scipy import linalg
import sys, pathlib, os
from ilex.data import average
from scipy.signal import correlate

# SHRINE imports
parent_folder = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.join(str(parent_folder), "src/SHRINE/"))
from dm_processing import get_kc, uncertainty_calc, get_ranges_above_max
from maximise_structure import (plot_DM_index, plot_noisy_I_DM_t, plot_I_at_max,
                                       plot_DCT_spectrum, plot_smooth_I_DM_t, plot_SP, 
                                       plot_detrended_noise, plot_relative_detrended_noise,
                                       plot_noise_at_max, plot_uncertainty, plot_relative_uncertainty,
                                       plot_adjusted_SP)

class _args:
    pass


def generate_profiles(ds, dt, tN, cfreq, bw, f0, dms):
    """
    Generate I matrix, a matrix of Stokes I time series for each trial DM.
    This function is an incoherent alternative to the native profile generation avaliable
    in SHRINE.

    Parameters
    ----------
    ds: 2D np.ndarray
        Stokes I dynamic spectrum
    dt: float
        sample resolution [ms]
    tN: int
        Downsampling factor in time
    cfreq: float
        Central frequency [MHz]
    bw: float
        Bandwidth [MHz]
    f0: float
        Reference frequency
    dms: 1D np.ndarray or array-like
        List of trial DMs
    
    Returns
    -------
    I: 2D np.ndarray
        I matrix (dm, t)
    """
    kdm = 4149.377593

    I = np.zeros((dms.size, ds.shape[1]//tN), dtype = float)

    # create freq array
    df = ds.shape[0] / bw
    freqs = np.linspace(cfreq + bw/2 - df/2, cfreq - bw/2 + df/2,
                        ds.shape[0])

    # Create copy of ds to dedisperse by rolling
    ds_dedis = ds.copy()

    # check if any channels are flagged
    goodchans = np.where(~np.isnan(ds[:, 0]))[0]

    # loop through trial dms and get dedispersed time series
    for i, dm in enumerate(dms):
        print(f"Searching trial DMs...   {i/dms.size:.2%}", end = "\r")

        # calculate dm shifts
        dmshifts = (kdm * dm * (1/freqs**2 - 1/f0**2)/(dt * 1e-3)).astype(int)

        # dedisperse by rolling
        for _, chan in enumerate(goodchans):
            ds_dedis[chan] = np.roll(ds[chan], -dmshifts[chan])
        
        # scrunch in frequency
        I[i] = average(np.mean(ds_dedis[goodchans], axis = 0), N = tN)
    print("Searching trial DMs...   100.00%")
    print("Trial DM profiles Generated")

    return I
            
    



def get_structure_maximised_dm(ds, dt: float, tN: int = 1, cfreq: float = 919.5, bw: float = 336, f0: float = 751.5, DMmin: float = -1.0, 
                                DMmax: float = 1.0, DMstep: float = 0.1, label: str = None, outdir: str = None, 
                                filter_order: int = 3, force_kc: int = None):
    """
    Taken from the _main() function in SHRINE/python/maximise_structure.py 
    and altered to be called as a function with the appropriate inputs.

    Parameters
    ----------
    ds: 2D np.ndarray
        Stokes I dynamic spectrum
    dt: float
        Time resolution in [ms]
    tN: int
        Downsampling factor in time
    cfreq: float
        Central frequency in [MHz], by default 919.5
    bw: float
        Bandwidth in [MHz], by default 336
    f0: float
        Reference frequency
    DMmin: float
        Minimum in trial DM range, by default -1.0
    DMmax: float
        Maximum in trial DM range, by default 1.0
    DMstep: float
        Step size in trial DM range, by default 0.1
    label: str
        Label for outputs, including figures and files
    outdir: str
        output directory for figures and files
    filter_order: int
        low pass filter spectral index
    force_kc: int
        manually force the k index cutoff for low pass filter, if None will estimate \n
        the best value, by default None
    
    """

    # make DM range and get I matrix 
    DM_data = np.arange(DMmin, DMmax, DMstep)
    I_data = generate_profiles(ds = ds, dt = dt, tN = tN, cfreq = cfreq, 
                                bw = bw, f0 = f0, dms = DM_data)
    
    # convert dt [ms -> us] to use rest of SHRINE functionality
    dt *= 1e3 * tN

    # Setup complete, start doing the math
    CI_data=dct(I_data, norm='ortho') #note "norm=ortho" to match MATLAB's dct
    dm_length,k_length=CI_data.shape

    print("Applying low-pass filter")
    # Low pass filter
    if force_kc is None:
        kc = get_kc(CI_data) #cutoff k index
    else:
        kc = force_kc
    O=filter_order #filter order
    k=np.linspace(1,k_length,k_length)
    # filter response
    fL=1/(1+(k/kc)**(2*O))

    # Pass DCT data through the combined filter to calculate structure parameter
    fL_diag=np.diag(fL) #make low-pass Filter into a diagonal matrix
    LPF_data=fL_diag@np.transpose(CI_data) #pass data through LPF
    I_smooth=idct(np.transpose(LPF_data), norm='ortho') #smooth data

    #1st derivative "high-pass" filter
    pi=np.pi
    hp=np.sqrt(2-2*np.cos((k-1)*pi/k_length)) #square root of the eigenvalues of D_1^T D_1

    #combined response
    filter=hp*fL
    filter_diag=np.diag(filter) #make Filter into a diagonal matrix

    print("Applying combined filter")
    #pass DCT data through the combined filter
    CI_filtered=filter_diag@np.transpose(CI_data)
    norm_CI_filtered=linalg.norm(CI_filtered, axis=0) #calculate structrue parameter, note numpy axis (see https://www.sharpsightlabs.com/blog/numpy-axes-explained/)

    print("calculating uncertainties")
    #Uncertainty calculations
    delta_I=I_data-I_smooth #de-trended noise

    #calculate uncertainty proper
    uncertainty = uncertainty_calc(delta_I, LPF_data, filter_diag)

    print("Finding Best DM")
    # Second round of uncertainty calculations
    max_index = np.argmax(norm_CI_filtered)
    delta_delta_I=delta_I-delta_I[max_index] 

    relative_uncertainty = uncertainty_calc(delta_delta_I, LPF_data, filter_diag)

    # Find uncertainty in structure maximizing DM
    max_structure_parameter = norm_CI_filtered[max_index]
    delta_DM_at_max = DM_data[max_index]

    adjusted_SPs = norm_CI_filtered + (norm_CI_filtered * relative_uncertainty) 

    possible_max_ranges = get_ranges_above_max(max_structure_parameter, adjusted_SPs)

    if len(possible_max_ranges) >= 1:
        if len(possible_max_ranges) == 1:
            one_range = True
        else:
            one_range = False
        # possible_max_ranges ALWAYS finds a minimum, so this is safe
        min_delta_DM = DM_data[possible_max_ranges[0][0]]
        if len(possible_max_ranges[-1]) == 2:
            max_delta_DM = DM_data[possible_max_ranges[-1][1]]
        else:
            max_delta_DM = None
    else:
        # Something has gone wrong if this runs
        # possible_max_ranges collects everything where `SP` >= `SP at max`
        # so at a minimum the max range should consist of `SP_at_max`
        min_delta_DM = None
        max_delta_DM = None
        one_range = None


    # save outputs and figures
    if outdir is None:
        outdir = os.getcwd()
    if label is None:
        label = "SMDM"

    print(f"Figures/Files will be saved to: {os.path.join(outdir, f'{label}*')}...")


    # data
    np.savetxt(os.path.join(outdir, f"{label}_SPs.dat"), norm_CI_filtered)
    np.savetxt(os.path.join(outdir, f"{label}_Uncertainties.dat"), uncertainty)
    np.savetxt(os.path.join(outdir, f"{label}_Relative_Uncertainties.dat"), relative_uncertainty)


    #Write a nice tidy file
    summary_file = open(os.path.join(outdir, f"{label}_structure_summaryfile.txt"), "w")
    summary_file.write(f"//begin maximise_structure summary//\n/*\n")
    summary_file.write(f"FRB Label: {label}\n")
    summary_file.write(f"Time Resolution: {dt}us\n")
    summary_file.write(f"kc: {kc}\n")
    if force_kc is None:
        summary_file.write(f"\tForced kc: {False}\n")
    else:
        summary_file.write(f"\tForced kc: {True}\n\t(This could be because of a provided force_kc value or because minimise_uncertainty was used.\n")
    summary_file.write(f"Structure Maximising Delta DM: {delta_DM_at_max}\n")
    summary_file.write(f"Uncertainty in Structure Maximising Delta DM: ")
    if min_delta_DM is not None:
        if min_delta_DM != np.min(DM_data):
            if max_delta_DM is not None:
                summary_file.write(f"{min_delta_DM-delta_DM_at_max}/+{max_delta_DM-delta_DM_at_max}\n")
            else:
                summary_file.write(f"{min_delta_DM-delta_DM_at_max}/+unknown\n")
                summary_file.write(f"\tUpper bound on uncertainty exceeds delta DM range.\n")
        else:
            if max_delta_DM is not None:
                summary_file.write(f"{min_delta_DM-delta_DM_at_max}/+{max_delta_DM-delta_DM_at_max}\n")
                summary_file.write(f"\tLower bound on uncertainty is equal to lower bound on delta DM range.\n")
                summary_file.write(f"\tThis probably means it is actually lower than {np.min(DM_data)}.\n")
            else:
                summary_file.write(f"{min_delta_DM-delta_DM_at_max}/+unknown\n")
                summary_file.write(f"\tLower bound on uncertainty is equal to lower bound on delta DM range.\n")
                summary_file.write(f"\tThis probably means it is actually lower than {np.min(DM_data)}.\n")
                summary_file.write(f"\tUpper bound on uncertainty exceeds delta DM range.\n")
    else:
        summary_file.write(f"-unknown/+unknown\n")
        summary_file.write(f"\tNo uncertainty range found. Something has gone wrong!\n")
    if not one_range:
        summary_file.write(f"\tUncertainty range was not continuous.\n")
        summary_file.write(f"\tRanges found were from:\n")
        for possible_range in possible_max_ranges:
            if len(possible_range) == 2:
                #good range
                summary_file.write(f"\t{DM_data[possible_range[0]]} to {DM_data[possible_range[1]]}\n")
            else:
                #bad range
                summary_file.write(f"\t{DM_data[possible_range[0]]} onwards. (Range finishes out of bounds).\n")
    summary_file.write(f"*/\n//end maximise_structure summary//\n\n\n")
    summary_file.close()


    # Save figures
    DM_file = open(os.path.join(outdir, f"DM.txt"), 'w')
    DM_file.write(f"{max_index}")
    DM_file.close()

    # set up args object 
    args = _args()
    args.dt = dt 
    args.label = os.path.join(outdir, label)

    # plots
    plot_DM_index(DM_data, args)

    # Intensity-time plots
    plot_I_at_max(I_data[max_index],I_smooth[max_index],args)
    plot_noise_at_max(delta_I[max_index], args)

    # I(DM,t) plots
    plot_noisy_I_DM_t(I_data, DM_data, args)
    plot_smooth_I_DM_t(I_smooth, DM_data, args)
    plot_detrended_noise(delta_I, DM_data, args)
    plot_relative_detrended_noise(delta_delta_I, DM_data, args)
    plot_DCT_spectrum(CI_data, args, kc)

    # Uncertainty-time plots
    plot_uncertainty(uncertainty, DM_data, args)
    plot_relative_uncertainty(relative_uncertainty, DM_data, args)

    # Structure parameter plots
    plot_SP(norm_CI_filtered, DM_data, args)
    plot_adjusted_SP(adjusted_SPs, DM_data, args)

    
    # print overall results
    print("\n Results for Structure maximising DM:")
    print(f"--"*20)
    summary_file = open(os.path.join(outdir, f"{label}_structure_summaryfile.txt"), "r").read()
    print(summary_file)

    return delta_DM_at_max



### NOT USED ###
def get_wmin_dm(ds, dt: float, tN: int = 1, cfreq: float = 919.5, bw: float = 336, f0: float = 751.5, DMmin: float = -1.0,
                DMmax: float = 1.0, DMstep: float = 0.1, label: str = None, outdir: str = None, filter_order: int = 3, 
                force_kc: int = None):

    """
    Find the DM that minimises the burst width.

    Function to maximise the S/N fluence in the burst to estimate DM. If we assume the noise the same regardless of DM,
    then we can repoise the problem of S/N fluence maximization to burst width minimisation.

    Parameters
    ----------
    ds: 2D np.ndarray
        Stokes I dynamic spectrum
    dt: float
        Time resolution in [ms]
    tN: int
        Downsampling factor in time
    cfreq: float
        Central frequency in [MHz], by default 919.5
    bw: float
        Bandwidth in [MHz], by default 336
    f0: float
        Reference frequency
    DMmin: float
        Minimum in trial DM range, by default -1.0
    DMmax: float
        Maximum in trial DM range, by default 1.0
    DMstep: float
        Step size in trial DM range, by default 0.1
    label: str
        Label for outputs, including figures and files
    outdir: str
        output directory for figures and files
    filter_order: int
        low pass filter spectral index
    force_kc: int
        manually force the k index cutoff for low pass filter, if None will estimate \n
        the best value, by default None
    
    """



    # make DM range and get I matrix 
    DM_data = np.arange(DMmin, DMmax, DMstep)
    I_data = generate_profiles(ds = ds, dt = dt, tN = tN, cfreq = cfreq, 
                                bw = bw, f0 = f0, dms = DM_data)

    nsamp = I_data.shape[1]
    
    # convert dt [ms -> us] to use rest of SHRINE functionality
    dt *= 1e3 * tN

    # Setup complete, start doing the math
    CI_data=dct(I_data, norm='ortho') #note "norm=ortho" to match MATLAB's dct
    dm_length,k_length=CI_data.shape

    print("Applying low-pass filter")
    # Low pass filter
    if force_kc is None:
        kc = get_kc(CI_data) #cutoff k index
    else:
        kc = force_kc
    O=filter_order #filter order
    k=np.linspace(1,k_length,k_length)
    # filter response
    fL=1/(1+(k/kc)**(2*O))

    # Pass DCT data through the combined filter to calculate structure parameter
    fL_diag=np.diag(fL) #make low-pass Filter into a diagonal matrix
    LPF_data=fL_diag@np.transpose(CI_data) #pass data through LPF
    I_smooth=idct(np.transpose(LPF_data), norm='ortho') #smooth data

    delta_I=I_data-I_smooth # De-trended noise
    # noise seems uniform across DMs, this is probably okay (?)
    constant_noise = np.min(I_smooth)
    noise_STD = np.std(delta_I)
    zeroed_I = I_smooth#-constant_noise
    
    # calculate fluence, assuming fluence is the same in all DM trials
    fluence = np.sum(zeroed_I[0])
    yfrac = 0.95
    startNstep = 100

    def find_N_length(tI, Nstart, Nstep):

        N = Nstart

        while True:
            corr = correlate(tI, 1/fluence * np.ones(N), mode = "valid")
            p = np.where(corr >= yfrac)[0]
            if p.size > 0:
                if N == 1:
                    print("The window offset found was 1, there may be something wrong with the data or input data is too small?")
                    return N, p
                if Nstep == 1:
                    # if p.size > 1:
                        # print("There appears to be two centroids to a minimum width.")
                    return N, p
                if Nstep > 1:
                    N, p = find_N_length(tI, N - Nstep, Nstep // 10)
                
                break
            N += Nstep

        return N, p

    sampwidth = np.zeros(DM_data.size, dtype = int)
    ps = []
    # sampstart = sampwidth.copy()
    for i, iDM in enumerate(zeroed_I):
        N = 1
        print(f"Searching for minimum burst width...   {i/DM_data.size:.2%}", end = "\r")
        sampwidth[i], pi = find_N_length(iDM, N, startNstep)
        ps += [pi]
    print(f"Searching for minimum burst width...   100.00%")
    
    # get minimum length in set of DM trials
    Nmin = np.argmin(sampwidth)

    # check if multiple occurences of Nmin are present
    multiNmin = False
    if np.sum(sampwidth == Nmin) > 1:
        multiNmin


    if outdir is None:
        outdir = os.getcwd()
    if label is None:
        label = "WMDM"


    # write summary file
    summary_file = open(os.path.join(outdir, f"{label}_minwidth_summaryfile.txt"), "w")
    summary_file.write("---Calculating DM to minimise burst width---\n")
    if multiNmin:
        summary_file.write(f"Multiple DM values give same minimum width, first occurence at DM = {DM_data[Nmin]} with a width of {Nmin * dt:.3f}us\n")
        summary_file.write(f"Other occurences: {DM_data[sampwidth == Nmin]}\n")
    else:
        summary_file.write(f"Minimum width of {Nmin * dt:.3f}us found at DM = {DM_data[Nmin]}\n")
    summary_file.write(f"Minimum width encompasses {yfrac:%} of burst fluence\n")
    summary_file.write(f"Uncertainty not YET IMPLEMENTED!!\n")
    summary_file.write(f"\n---End of Summary file---")
    summary_file.close()



    print("Making figures")
    # make plots
    ## plot best I
    plt.figure(figsize = (10,8))
    x = np.linspace(0, (nsamp-1)*dt*1e-3, nsamp)
    plt.plot(x, I_data[Nmin], 'k', linewidth = 1.0, alpha = 0.3)
    plt.plot(x, zeroed_I[Nmin], 'k', linewidth = 1.5)
    ylim = plt.gca().get_ylim()
    # plot bounds
    plt.plot([ps[Nmin][0] * dt * 1e-3]*2, ylim, 'r--')
    plt.plot([(ps[Nmin][0] + sampwidth[Nmin])*dt*1e-3]*2, ylim, 'r--')
    plt.gca().set_ylim(ylim)
    plt.xlabel("Time [ms]", fontsize = 16)
    plt.ylabel("Flux (arb.)", fontsize = 16)
    plt.grid(axis = 'both')
    plt.gcf().tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_best_I.png"))
    plt.close()


    ## plot fluence vs DM trial
    plt.figure(figsize = (10,10))
    plt.plot(DM_data, np.sum(zeroed_I, axis = 1), 'k', linewidth = 1.5)
    plt.grid(axis = 'both')
    plt.xlabel("DM [$pc cm^{-3}$]", fontsize = 16)
    plt.ylabel("Fluence (arb.)", fontsize = 16)
    plt.title("Fluence of smoothed data")
    plt.gcf().tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_fluence_DM.png"))
    plt.close()


    ## plot burst width vs DM
    plt.figure(figsize = (10,10))
    plt.plot(DM_data, dt*np.asarray(sampwidth, dtype = float), 'k', linewidth = 1.5)
    plt.grid(axis = 'both')
    plt.xlabel("DM [$pc cm^{-3}$]", fontsize = 16)
    plt.ylabel("Burst width [us]", fontsize = 16)
    plt.gcf().tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_width_DM.png"))
    plt.close()


    # print overall results
    print("\n Results for Structure maximising DM:")
    print(f"--"*20)
    summary_file = open(os.path.join(outdir, f"{label}_minwidth_summaryfile.txt"), "r").read()
    print(summary_file)

    return DM_data[Nmin]