##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 25/09/2023 
##
##
## 
## 
## Library of functions for HTR processing (coherent)
## 
## 
##
##===============================================##
##===============================================##
# imports
import numpy as np
from scipy.fft import fft, ifft, next_fast_len
import sys, os
from .data import rotate_data, average
from math import ceil



##=====================##
## AUXILIARY FUNCTIONS ##
##=====================##

def phasor_DM(f, DM: float, f0: float):
    """
    Info:
        Apply Phasor rotation based on DM, coherent Dispersion
        This rotation can be applied 

    Args:
        f (ndarray): frequency array [MHz]
        DM (float): Dispersion Measure [pc/cm3]
        f0 (float): reference frequency [rad]

    Returns:
        p (ndarray): Phasor np.exp()
    
    """
    # constants
    kDM = 4.14938e3         # DM constant

    return np.exp(2j*np.pi*kDM*DM*(f-f0)**2/(f*f0**2)*1e6)



def phasor_(f, tau: float, phi: float):
    """
    Info:
        Apply A generalized phasor rotation -> 

    Args:
        f (ndarray): frequency array [MHz]
        tau (float): Time delay [us]
        phi (float): Phase delay [rad]  

    Returns:
        p (ndarray): Phasor np.exp()
    
    """
    return np.exp(2j*np.pi*tau*f + 1j*phi)
    


def mem_fft(x, fft_len, _map = 'r+'):
    """
    Info:
        Apply FFT and save to temp memory map

    Args:
        x (ndarray): data to fft
        fft_len (int): len of fft array
        _map (str): mapper, 'r' for read only or 'r+' read and writing.
                    see mmap_mode in numpy.load()

    Returns:
        xfft (memmap): memory map of fft

    """
    xfft = fft(x, fft_len)

    # save to temp array
    np.save("temp_fft.npy", xfft)

    # load temp array as memory map
    xfft = np.load("temp_fft.npy", mmap_mode = _map)


    return xfft




def mem_ifft(x, fft_len, dat_len, _map = 'r+'):
    """
    Info:
        Apply iFFT and save to temp memory map

    Args:
        x (ndarray): data to fft
        fft_len (int): len of fft array
        dat_len (int): initial lenght of array
        _map (str): mapper, 'r' for read only or 'r+' read and writing.
                    see mmap_mode in numpy.load()

    Returns:
        xfft (memmap): memory map of ifft

    """
    xifft = ifft(x, fft_len)[:dat_len]

    # save to temp array file
    np.save("temp_ifft.npy", xifft)

    # load temp array as memory map
    xifft = np.load("temp_ifft.npy", mmap_mode = _map)


    return xifft


def arr2mmap(x, filename, _map = 'r+'):
    """
    Info:
        Convert array to memory map for memory efficiency

    Args:
        x (ndarray): data to convert to memory map
        filename (str): filename to store memory map
        _map (str): mapper, 'r' for read only or 'r+' read and writing.
                    see mmap_mode in numpy.load()

    Returns:
        xm (memmap): memory map of x data array
    
    """

    # save to temp array file
    np.save(filename, x)

    # load temp array as memory map
    xm = np.load(filename, mmap_mode = _map)

    return xm



def rm_mmap_file(x):
    """
    Info:
        Remove Memory map file

    Args:
        x (memmap): memory map object

    """

    os.remove(x.filename)

    return







##==================##
## STOKES FUNCTIONS ##
##==================##

def stk_I(X, Y):
    """
    Info:
        Produces stokes I data

    Args:
        X (ndarray): X polarisation data
        Y (ndarray): Y polarisation data

    Returns:
        S (ndarray): Stokes data

    """

    return np.abs(X)**2 + np.abs(Y)**2



def stk_Q(X, Y):
    """
    Info:
        Produces stokes Q data

    Args:
        X (ndarray): X polarisation data
        Y (ndarray): Y polarisation data

    Returns:
        S (ndarray): Stokes data
        
    """

    return np.abs(X)**2 - np.abs(Y)**2

def stk_U(X, Y):
    """
    Info:
        Produces stokes U data

    Args:
        X (ndarray): X polarisation data
        Y (ndarray): Y polarisation data

    Returns:
        S (ndarray): Stokes data
        
    """

    return 2 * np.real(np.conj(X) * Y)

def stk_V(X, Y):
    """
    Info:
        Produces stokes V data

    Args:
        X (ndarray): X polarisation data
        Y (ndarray): Y polarisation data

    Returns:
        S (ndarray): Stokes data
        
    """

    return 2 * np.imag(np.conj(X) * Y)

## array of stokes functions ##
STK_FUNC = {"I":stk_I, "Q":stk_Q, "U":stk_U, "V":stk_V}









## Additional processing functions ##
def baseline_correction(ds, sigma: float = 5.0, guard: float = 1.0, 
                        baseline: float = 50.0, rmsmp: float = 0.5,
                        tN: int = 50, dt: float = 0.001,
                        rbounds = None):
    """
    Info:
        Apply baseline corrections to the Dynamic spectra data

    Args:
        ds (ndarray): Dynamic spectra
        sigma (float): S/N threshold for bounds
        guard (float): Time in [ms] between rough bounds and rms crop region
        baseline (float): Width of buffer in [ms] to estimate baseline
                          correction
        rmsmp (float): Phase difference between maximum point in time and mid
                       point of rms crop used to estimate rough initial S/N
        tN (int): Time Averaging factor for Dynamic spectrum, helps with
                    S/N calculation.
        dt (float): Time resolution in [ms] of dynamic spectra
        rbounds (list): Bounds of FRB burst, if Unspecified, the code will do a rough S/N
                        calculation to determine a bursts bounds
        norm (bool): Apply baseline correction to dynamic spectra

    Returns:
        ds (ndarray): Baseline corrected (zero-mean, unit variance) dynamic spectra 
        bs_mean (ndarray): Baseline mean
        bs_std (ndarray): Baseline std
        rbounds (ndarray): Bounds of FRB burst in Phase units

    """
    dt *= ds.shape[0]/336.0

    # ms -> ds time bin converters
    get_units_avg = lambda t : int(ceil(t/(dt * tN)))
    get_units = lambda t : int(ceil(t/dt))

    if rbounds is None:
        ## Rough normalize 
        ds_r = average(ds, axis = 1, N = tN)
        rmean = np.mean(ds_r, axis = 1)
        rstd = np.std(ds_r, axis = 1)

        ds_rn = ds_r - rmean[:, None]
        ds_rn /= rstd[:, None]

        
        ## find burst bounds
        print("Looking for bounds of burst...")
        # get peak, crop rms and do rough S/N calculation
        t_rn = np.mean(ds_rn, axis = 0)
        peak = np.argmax(t_rn)
        print(f"Peak found at phase of {peak / t_rn.size}")

        rms_w = get_units_avg(baseline)
        rms_crop = np.roll(t_rn, int(rmsmp * ds_rn.shape[1]))[peak-rms_w:peak+rms_w]
        rms = np.mean(rms_crop**2)**0.5

        # calculate S/N
        t_sn = t_rn / rms
        rbounds = np.argwhere(t_sn >= sigma)[[0,-1]]/t_sn.size
        rbounds = np.asarray((rbounds*ds.shape[1]), dtype = int)[:,0]
    

    print(f"Phase bounds of burst: {rbounds/ds.shape[1]}")
    ## calculate baseline corrections
    guard_w = get_units(guard)
    rms_w = get_units(baseline)
    lhs_crop = ds[:,rbounds[0]-guard_w-rms_w:rbounds[0]-guard_w]
    rhs_crop = ds[:,rbounds[1]+guard_w:rbounds[1]+guard_w+rms_w]
    bl_crop = np.concatenate((lhs_crop, rhs_crop), axis = 1)

    bs_mean = np.mean(bl_crop, axis = 1)
    bs_std = np.std(bl_crop, axis = 1)


    ## re-normalize (Baseline Correction)
    print("Applying Baseline corrections...")
    ds -= bs_mean[:, None]
    ds /= bs_std[:, None]

    return ds, rbounds



















# stft htr function
def make_stokes(xpol, ypol, stokes = "I", nFFT = 336, negateQ = True, nworkers = 4, BLOCK_SIZE = 200e6,
                BIT_SIZE = 8):
    """
    Info:
        Performs the stft on the xpol and ypol htr products

    Args:
        xpol (ndarray): X polarisation time series (ideally a memory map)
        ypol (ndarray): Y polarisation time series (ideally a memory map)
        stokes (str): Stokes product type ["I", "Q", "U", "V"]
        nFFT (int): FFT window size (number of channels for dynamic spectrum)
        nworkers (int): number of "workers" for parallel processing (see scipy.fft.fft)
        BLOCK_SIZE (int): maximum memory allocated at a time when running fft
        conv (str): Stokes sign convention, ["celebi", "straten"]

    Returns:
        dynspec (ndarray): Final dynamic spectrum

    """

    prog_str = f"""[Stokes] = {stokes} with [nFFT] = {nFFT}:    [BLOCK SIZE] = {BLOCK_SIZE/1e6:.3f} MB with [BIT SIZE] = {BIT_SIZE} Bits"""

    if stokes not in "IQUV":
        print("Invalid Stokes")
        return 


    # First need to chop data so that an integer number of FFT windows can
    # be performed. Afterwards, this data will be split into coarse BLOCKS
    # with specific memory constraints. 

    # define parameters
    nsamps  = xpol.size                  # original number of samples in loaded dataset
    fnsamps = (nsamps // nFFT) * nFFT    # number of samples after chopping 
    nwind   = fnsamps // nFFT            # number of fft windows along time series

    # memeory block paramters
    nwinb = int(BLOCK_SIZE // (nFFT * BIT_SIZE))    # num windows in BLOCK
    nsinb = int(nwinb * nFFT)                     # num samples in BLOCK
    nblock= int(nwind // nwinb)                     # num BLOCKS

    # create empty array
    ds = np.zeros((nFFT, nwind), dtype = np.float32)

    # block array for looping over data
    b_arr = np.empty((0,2), dtype = int)
    for i in range(nblock):
        b_arr = np.append(b_arr, [[i*nwinb,(i+1)*nwinb]], axis = 0)
    # append extra block at end
    if nblock*nsinb < nsamps:
        b_arr = np.append(b_arr, [[(i+1)*nwinb,nwind]], axis = 0)

    # loop over blocks
    for i, b in enumerate(b_arr): # b is bounds of block in nFFT windows
        sb = b * nFFT
        wind_w = b[1] - b[0]
        ds[:,b[0]:b[1]] = STK_FUNC[stokes](fft(xpol[sb[0]:sb[1]].copy().reshape(wind_w, nFFT), axis = 1),
                                 fft(ypol[sb[0]:sb[1]].copy().reshape(wind_w, nFFT), axis = 1)).T
        
        # print progress
        print(f"[MAKING DYNSPEC]:    [Progress] = {(i+1)/(nblock+1)*100:3.3f}%:    " + prog_str,
              end = '\r')

    print("[MAKING DYNSPEC]:    [Progress] = 100.00%:    " + prog_str + "\n")   

    print("Dynamic spectrum created with:")
    print(f"{ds.shape[0]} channels")
    print(f"{ds.shape[1]} time samples")


    # negate Q if nessesary
    if negateQ and stokes == "Q":
        ds *= -1

    return ds
    














def coherent_desperse(t, cfreq, bw, f0, DM, fast = False, 
                      DM_iter = 50):
    """
    Info:
        Function that coherently dedisperses X,Y htr data products in (f) domain,
        It is assumed that frequencies start from the top of the band.

    Args:
        t (time series): time series
        cfreq (float): central frequency of band
        bw (float): bandwidth
        f0 (float): refernece frequency
        DM (float): Dispersion Measure
        fast (bool): Apply padding to significantly speed up function at the cost of a small precision error
        DM_iter (int): Reduces the amount of memory used when applying dispersion at the cost of a little performance


    Returns:
        t_d (ndarray): time series despersed


    """

    prog_str = f"[bw] = {bw} MHz with [cfreq] = {cfreq} MHz:    [DM] = {DM} pc/cm3 at a ref freq [f0] = {f0} MHz"

    # constants
    kDM = 4.14938e3         # DM constant

    next_len = t.size
    if fast:
        next_len = next_fast_len(next_len)

    # ifft
    t_d = fft(t, next_len)

    # apply dispersion
    uband = cfreq + bw/2                            # upper band
    bw = bw*(next_len/t.size)                       # updated bandwidth due to zero padding in time domain

    BLOCK_size = int(next_len / DM_iter)            # number of samples per DM iteration
    BLOCK_end = next_len - BLOCK_size * DM_iter     # in case number of samples don't divide into DM_iter evenly, usually
                                                    # a handful of samples are left at the end, this is BLOCK_end
    BLOCK_bw = float(bw*BLOCK_size/next_len)        # amount of bandwidth being covered per DM iteration

    # iterate over chuncks to save memory
    for i in range(DM_iter):
        freqs = (np.linspace(uband - i*BLOCK_bw, uband - (i+1)*BLOCK_bw, BLOCK_size, endpoint = False) 
                +bw/next_len/2)

        # disperse part of t series
        t_d[i*BLOCK_size:(i+1)*BLOCK_size] *= phasor_DM(freqs, DM, f0)

        print(f"[DISPERSING]:    [Progress] = {(i+1)/(DM_iter+1)*100:3.3f}%:    " + prog_str, end = "\r")

    # apply last chunck if needed
    if BLOCK_end > 0:
        freqs = np.linspace(uband - (i+1)*BLOCK_bw,uband - bw, BLOCK_end, endpoint = False) +bw/next_len/2

        # disperse
        t_d[(i+1)*BLOCK_size:] *= phasor_DM(freqs, DM, f0)

    print("[DISPERSING]:    [Progress] = 100.00%:    " + prog_str + "\n")

    # inverse fourier tranform back to time domain
    t_d = ifft(t_d, next_len)[:t.size]

    return t_d















def pulse_fold(ds, MJD0, MJD1, F0, F1, sphase = None):
    """
    Info:
        Takes Pulsar dynamic spectrum and folds it, removes periods
        at far left and right sides to avoid band artifacts produced during
        de-dispersion.

    Args:
        ds (ndarray): dynamic spectrum
        MJD0 (float): Initial Epoch MJD
        MJD1 (float): Observation MJD
        F0 (float): initial Epoch Frequency period
        F1 (float): Spin-down rate
        sphase (float): Starting phase of folding, if not given
                        will be estimated (best done using "I" ds)

    Returns:
        ds_f (ndarray): Raw folded Dynamic spectra
        sphase (float): Starting phase of folding from original dynspec

    """
    print("Pulse Folding Dynspec...")
    # normalise needed to help find fold bounds
    ds_mean = np.mean(ds, axis = 1)[:, None]
    ds_std = np.std(ds, axis = 1)[:, None]

    ## Calculate Period T in [s]
    T = 1/(F0 - F1 * (MJD1 - MJD0)*86400)
    print(f"with period T = {T}")
    dt = 1e-6 * (ds.shape[0]/336) # get time resolution of dynspec

    ## Fold dynamic spectra
    fold_w = int(T / dt)          # fold width in samples (assumed dt = 1 us)
    fold_n = int(ds.shape[1]/fold_w)     # number of folds

    # find index of peak in second period, then get starting phase
    if sphase is None:
        search_crop = ds[:,fold_w:2*fold_w]
        search_crop = (search_crop - ds_mean)/ds_std
        pulse2i = np.mean(search_crop, axis = 0).argmax()
        pulse2i += fold_w
        sphase = pulse2i - int(fold_w/2)
    else:
        # put in phase units
        sphase = int(sphase*ds.shape[1])


    # reshape to average folds together
    # ignore side periods due to dedispersing
    ds_r = ds[:,sphase:sphase + fold_w * (fold_n-2)].copy()
    ds_f = np.mean(ds_r.reshape(ds_r.shape[0], (fold_n-2), fold_w), axis = 1)

    
    return ds_f, sphase / ds.shape[1]













    







def coherent_scatter():
    """
    
    """
    pass





def coherent_rotate():
    """

    """
    pass













# ##==========================================##
# ## Master function to update ASKAP X, Y POL ##
# ## data products                            ##
# ##==========================================##

# def update_XY(X, Y, filename, cfreq: float = 0.0, bw: float = 336.0, f0: float = None,
#               DM: float = 0.0, tau: float = 0.0, phi: 
#               float = 0.0, psi: float = 0.0, _iter: int = 50, fast: bool = True, save_XY = True):
#     """
#     Make new X,Y data products with the following changes:

#     del DM:         apply a small change in dispersion coherently
#     tau:            apply a time delay coherently
#     phi:            apply a phase delay coherently
#     psi:            apply a rotational offset between X and Y

#     This function can be memory intensive, especially if del RM is being
#     used. To be safe a minimum of 24GB RAM (or ~3x sizeof(X)) is nessesary to use all of the 
#     above features.

#     safe mem usage ~3x sizeof(X, Y)


#     ##==== inputs ====##
#     DM (pc/cm3):             Dispersion Measure
#     tau (us):                Time delay (from POLCAL)
#     phi (rad):               Phase delay (from POLCAL)
#                              Note: tau and phi derived from POLCAL.py
#                              are what has already been applyied to X, Y, to
#                              correct for these in this function they need to be
#                              negated.
#     psi (rad):               Rotation offset (from POLCAL)
#     filename:                Common prefix for X, Y filenames
#                              i.e. [{filename}_X.npy]
#     fast:                    Apply Zero-padding to speed up FFT/iFFT
#     X:                       X pol memory map
#     Y:                       Y pol memory map


#     ##==== outputs ====##  
#     None:           This function does not have any outputs

#     """
#     print("Making new X and Y pol products using the following:")
#     print("----------------------------------------------------")
#     print(f"delta DM:    {DM:.4f}  (pc/cm3)")
#     print(f"tau:         {tau:.4g}  (us)")
#     print(f"phi:         {phi:.4f}  (rad)")
#     print(f"psi:         {psi:.4f}  (rad)\n")

#     t1 = time()
#     ##================================##
#     ## APPLY DM AND TIME/PHASE DELAYS ##
#     ##================================##
#     mmap_flag = type(X) == "numpy.memmap"

#     if mmap_flag:
#         print("Loading X,Y data products")
#     else:
#         print("Passing X,Y data products")

#     # output filenames
#     X_ofile = f"{filename}_X.npy"
#     Y_ofile = f"{filename}_Y.npy"


#     # process flags
#     DM_flag = DM != 0.0                         # DM flag
#     leak_flag = (tau != 0.0 or phi != 0.0)      # leakage flag
#     psi_flag = psi != 0.0                       # rotation offset flag


#     # get new len for FFT
#     fft_len = X.size        # size of fft for optimal speed
#     dat_len = X.size        # size of original data set (X or Y)
#     if fast:
#         fft_len = next_fast_len(X.size)


#     # set up for iterative process
#     uband = cfreq + bw/2                            # upper band
#     bw = bw*(fft_len/dat_len)                       # updated bandwidth due to zero padding in time domain

#     BLOCK_size = int(fft_len / _iter)               # number of samples per DM iteration
#     BLOCK_end = fft_len - BLOCK_size * _iter        # in case number of samples don't divide into DM_iter evenly, usually
#                                                     # a handful of samples are left at the end, this is BLOCK_end
#     BLOCK_bw = float(bw*BLOCK_size/fft_len)         # amount of bandwidth being covered per DM iteration


#     if f0 is None:          # reference frequency
#         f0 = cfreq


#     # make Function to retrieve a block of frequeuncy band
#     def get_fblock(_i):
#         """
#         Get block of bandwidth (for Memory efficieny)
#         """
#         if _i < _iter:
#             _f = (np.linspace(uband - _i*BLOCK_bw, uband - (_i+1)*BLOCK_bw, BLOCK_size, endpoint = False) 
#                 +bw/fft_len/2)
#         elif i == _iter:
#             _f = (np.linspace(uband - (_i+1)*BLOCK_bw,uband - bw, BLOCK_end, endpoint = False) 
#                 +bw/fft_len/2)
#         else:
#             print("Something went wrong: asking for chunk of frequency outside bandwidth")
#             sys.exit()
#         return _f



#     ##=====================================##
#     ## APPLY PROCESSES WITHOUT NEEDED FFTS ##
#     ##=====================================##

#     ## apply rotational offset
#     if psi_flag:
#         print("Applying rotational offset...")
#         X, Y = rotate_data(X, Y, psi)
#     else:
#         print("Skipping Rotation offset [psi = 0.0]")
    


#     ##==============================##
#     ## APPLY PROCESSES NEEDING FFTS ##
#     ##==============================##

#     # check whether X pol needs to be FFT'd
#     if DM_flag or leak_flag or fday_flag:
#         # FFT X pol
#         print("Applying FFT to [X] pol")
#         X = fft(X, fft_len)
        

#     if DM_flag or fday_flag:
#         # FFT Y pol
#         print("Applying FFT to [Y] pol")
#         Y = fft(Y, fft_len)
        


#     ## APPLY DM AND LEAKAGE
#     if DM_flag or leak_flag:
#         # iterate, this iterates over blocks of the array at a time,
#         # more memory efficient at the cost of a little performance
#         print(f"Applying DM = {DM_flag}")
#         print(f"Applying Leakage = {leak_flag}")
        
#         # do iteration
#         for i in range(_iter):
#             print(f"progress: {i/_iter*100:.2f} %", end = '\r')
#             freqs = get_fblock(i)

#             # dedisperse rotate
#             if DM_flag:
#                 X[i*BLOCK_size:(i+1)*BLOCK_size] *= phasor_DM(freqs, DM, f0)
#                 Y[i*BLOCK_size:(i+1)*BLOCK_size] *= phasor_DM(freqs, DM, f0)
            
#             # leakage rotate
#             if leak_flag:
#                 X[i*BLOCK_size:(i+1)*BLOCK_size] *= phasor_(freqs, tau, phi)
            
#         # apply last block if needed (consequency of doing integer iteration)
#         print(f"progress: 100.00 %\n")
#         if BLOCK_end > 0:
#             freqs = get_fblock(i+1)

#             # dedisperse rotate
#             if DM_flag:
#                 X[(i+1)*BLOCK_size:] *= phasor_DM(freqs, DM, f0)
#                 Y[(i+1)*BLOCK_size:] *= phasor_DM(freqs, DM, f0)
            
#             # leakage rotate
#             if leak_flag:
#                 X[(i+1)*BLOCK_size:] *= phasor_(freqs, tau, phi)
        


#     else:
#         print(f"Skipping Despersion and leakage correction [DM = {DM:.1f}], [tau = {tau:.1f}], [phi = {phi:.1f}")

#     # # free up memory
#     # if DM_flag or leak_flag:
#     #     X = arr2mmap(X, X_ofile)
#     # if DM_flag:
#     #     Y = arr2mmap(Y, Y_ofile)


#     # check whether X pol needs to be FFT'd
#     if DM_flag or leak_flag:
#         # FFT X pol
#         print("Applying iFFT to [X] pol")
        
#         X = ifft(X, fft_len)[:dat_len]
        
#     if DM_flag:
#         # FFT Y pol
#         print("Applying iFFT to [Y] pol")
        
#         Y = ifft(X, fft_len)[:dat_len]
        
#     print(f"Execution Time: {time() - t1:.2f} s")
    
#     # due to using memory maps with the mmap_mode = 'r+' option,
#     # data is already stored in our output files.
#     print("Completed!")

#     if save_XY:
#         np.save(X_ofile, X)
#         print(f"[X] pol saved as: {X_ofile}")

#         np.save(Y_ofile, Y)
#         print(f"[Y] pol saved as: {Y_ofile}")
#         return
    
#     else:
#         print("removing temp files...")
#         os.remove(X_ofile)
#         os.remove(Y_ofile)
#         return X, Y














# def make_stokes(X, Y, fname, stk = "", nFFT = 336, tN = 1, fN = 1, t_crop = None, f_crop = None,
#                 nwork = 1, BLK_S = 200e6, BIT_S = 8):
#     """
#     Make Stokes spectra from X and Y pol products Using CELEBI stokes conventions. 
#     Can also apply a handful of processes to the dynspec products.

#     process:
#         1. Make X,Y Dynamic Spectra
#         2. Apply processes -> Dedispersion
#                            -> Baseline correction
#         3. mean/std corrections
#         4. Make IQUV stokes dynamic spectra
#         5. Apply post-processes -> Time/Frequency scrunching
#                                 -> Time/Frequency Cropping
    

#     ##==== inputs ====##
#     X:              X polarisation time series filename (~3ns)
#     Y:              Y polarisation time series filename (~3ns)
#     stk:            Stokes dynspec to make [pass as string i.e. "QUV" for all or "Q" for Q]
#                     Stokes I dynamic spectra will always be created. To build just stokes I, 
#                     pass an empty string.
#     nFFT:           Number of channels for dynamic spectra
#     fname:          Common Filename Prefix for dynamic spectra

#     DM [pc/cm3]:    Dispersion Measure 
#     f0 [rad]:       Reference Frequency
#     cfreq [MHz]:    Central Frequency of band
#     bw [MHz]:       Bandwidth

#     tN:             Time averaging Factor
#     fN:             Frequency averaging Factor
#     t_crop:         Time Axis Crop of dynspec [taken as a phase between 0-1]
#     f_crop:         Frequeny Axis Crop of dynspec [taken as a phase between 0-1]

#     nwork:          Number of workers for STFT (FFT)
#     BLK_S [B]:     size of BLOCK to split time series into (More Memory efficient)
#     BIT_S [B]:      size of data in bytes

#     ##==== outputs ====##
#     None:
    
#     """
#     t1 = time()
#     print(f"Making Stokes dynspec: {stk} with {nFFT:.0f} f channels")
#     print("Applying the following processes:")

    
#     stk = "I" + stk                         # update stokes string with I 
#     pol = {"X":X, "Y":Y}                    # convienient container 


#     ## Set up iteration ##
#     # First need to chop data so that an integer number of FFT windows can
#     # be performed. Afterwards, this data will be split into coarse BLOCKS
#     # with specific memory constraints. 

#     # define parameters
#     nsamps  = pol['X'].size                        # original number of samples in loaded dataset
#     fnsamps = (nsamps // nFFT) * nFFT       # number of samples after chopping 
#     nwind   = fnsamps // nFFT               # number of fft windows along time series

#     # memeory block paramters
#     nwinb = int(BLK_S // (nFFT * BIT_S))    # num windows in BLOCK
#     nsinb = int(nwinb * nFFT)               # num samples in BLOCK
#     nblock= int(nwind // nwinb)             # num BLOCKS

#     # crop block parameters
#     sblock= 0                               # block where crop starts
#     eblock= nblock                          # block where crop ends
#     ssamp = 0                               # sample number where crop starts
#     esamp = fnsamps                         # sample number where crop ends
#     schan = 0                               # channel number crop stars
#     echan = nFFT                            # channel number crop ends
#     nchan = nFFT                            # number of channels
#     nblockc = nblock                        # number of blocks in crop
#     swinc = 0                               # sample FFT where crop starts
#     ewinc = nwind                           # sample FFT where crop ends

#     if t_crop is not None:  # update time crop
#         swinc = int(t_crop[0]*nwind)
#         ewinc = int(t_crop[1]*nwind)
#         sblock = int(t_crop[0]*nblock)
#         eblock = int(t_crop[1]*nblock)
#         nblockc = (ewinc - swinc) // nwinb
    
#     if f_crop is not None: # update frequency crop
#         schan = int(f_crop[0]*nFFT)
#         echan = int(f_crop[1]*nFFT)
#         nchan = echan - schan

#     print("Will Build Stokes Dynamic spectra with the following parameters:")
#     print(f"Time crop: {t_crop}")
#     print(f"Frequency crop: {f_crop}")
#     print(f"# channels: {nchan}")
#     print(f"# samples: {ewinc - swinc}")


#     # build block arrays, sample crop may not nessesarily align with blocking
#     # also, We want to conserve memory, it wouldn't make sense to store part
#     # of the stokes dynamic spectra outside of the crop. In addition, we also
#     # want to reduce computation as much as possible, that is, only do one FFT
#     # for each X and Y polarisation. To Do this we will incorporate a handful of 
#     # logic:
#     #       -> Only store FFT of crop
#     #       -> Save FFT(X) and FFT(Y) crop to temporary array file .npy (for memory mapping)


#     # First, build block array that will handle outside the sample crop
#     outarr = np.empty((0,2), dtype = int)                               # initialise empty arrays
#     inarr  = np.empty((0,2), dtype = int)

#     outarrL = np.empty((0,2), dtype = int)
#     for i in range(0, sblock):      # LHS outside crop
#         outarrL = np.append(outarrL,[[i*nwinb, (i+1)*nwinb]], axis = 0)
#     if swinc > 0:      # if first window not start bound
#         outarrL = np.append(outarrL, [[sblock*nwinb, swinc]], axis = 0)

#     outarrR = np.empty((0,2), dtype = int)        # RHS outside crop
#     for i in range(eblock+1, nblock):
#         outarrR = np.append(outarrR, [[i*nwinb, (i+1)*nwinb]], axis = 0)
#     if ewinc < nwind - 1:
#         outarrR = np.append([[ewinc, (eblock+1)*nwinb]], outarrR, axis = 0)

#     # combine L and R sides to form single array
#     outarr = np.append(outarrL, outarrR, axis = 0)

#     # now, build block array that will handle inside the sample crop
#     i = -1
#     for i in range(nblockc):
#         inarr = np.append(inarr, [[swinc + i*nwinb, swinc + (i+1)*nwinb]], axis = 0)
#     inarr = np.append(inarr, [[swinc + (i+1)*nwinb, ewinc]], axis = 0)


#     ## FFT arguments
#     fftp = {"workers": nwork, "axis": 1}




#     ## auxillary function to process FFT'd data ##
#     def proc_fft(x):
#         """
#         Process fft:
#         - despersion


#         """

#         return x




#     ##===========================##
#     ## PROCESS DATA OUTSIDE CROP ##
#     ##===========================##
#     # initialise mean and stds
#     std = {}                             # std of stokes dynspec squared -> sqrt taken at end
#     meanI = np.zeros(nchan)              # mean of I dynspec
#     for S in stk:
#         std[S] = np.zeros(nchan)


#     # loop outarr
#     print("Processing data outside crop...")
#     for i, blk in enumerate(outarr):
#         print(f"progress: {i/outarr.shape[0]*100:.2f} %", end = '\r')
#         # perform FFT on X and Y
#         blk_len = blk[1] - blk[0]
#         samp_i = blk * nFFT
#         Xfft = fft(pol['X'][samp_i[0]:samp_i[1]].copy().reshape(blk_len, nFFT),
#                    **fftp).T[schan:echan]
#         Yfft = fft(pol['Y'][samp_i[0]:samp_i[1]].copy().reshape(blk_len, nFFT),
#                    **fftp).T[schan:echan]

#         # process fft data
#         Xfft = proc_fft(Xfft)
#         Yfft = proc_fft(Yfft)

#         # loop over stokes
#         for S in stk:
#             # get stokes block whilst cropping in frequency
#             stk_blk = STK_FUNC[S](Xfft, Yfft)

#             # add to std
#             std[S] += np.sum(stk_blk**2, axis = 1)

#             # add to mean
#             if S == "I":
#                 meanI += np.sum(stk_blk, axis = 1)
            
#     print("progess: 100.00%\n")
            
    

#     ##==========================##
#     ## PROCESS DATA INSIDE CROP ##
#     ##==========================##
#     # initialize crop of POL data
#     print(inarr)
#     P_crop = np.zeros((nchan, ewinc - swinc), dtype = "complex64")
#     for P in "XY":
        
#         # loop over blocks
#         print(f"Processing {P} pol data inside crop...")
#         for i, blk in enumerate(inarr):
#             print(f"progress: {i/inarr.shape[0]*100:.2f} %", end = '\r')
#             # perform FFT
#             blk_len = blk[1] - blk[0]
#             crop_i = blk - swinc        # crop indexing
#             samp_i = blk * nFFT         # pol data indexing

#             fft_blk = fft(pol[P][samp_i[0]:samp_i[1]].copy().reshape(blk_len, nFFT), 
#                                                 **fftp).T[schan:echan]

#             # process fft data and store in array
#             P_crop[:,crop_i[0]:crop_i[1]] = proc_fft(fft_blk)


#         print("progress: 100.00 %\n")
#         # save crop of P to temporary array
#         np.save(f"temp_{P}fft.npy", P_crop)
    
#     # release memory
#     del P_crop
#     del fft_blk



#     ##=======================##
#     ## BUILD STOKES DYNSPECS ##
#     ##=======================##
#     S_crop = np.zeros((nchan, ewinc - swinc), dtype = float)
#     # load in temp pol arrays
#     pol['X'] = np.load("temp_Xfft.npy", mmap_mode = 'r')
#     pol['Y'] = np.load("temp_Yfft.npy", mmap_mode = 'r')

#     for S in stk:
#         # loop over inarr
#         print(f"Processing stokes {S} dynamic spectrum...")
#         for i, blk in enumerate(inarr):
#             print(f"progress: {i/inarr.shape[0]*100:.2f} %", end = '\r')
#             crop_i = blk - swinc        # crop indexing
#             # perform stokes calc
#             stk_blk = STK_FUNC[S](pol['X'][:,crop_i[0]:crop_i[1]],
#                                                      pol['Y'][:,crop_i[0]:crop_i[1]])

#             # update std
#             std[S] += np.sum(stk_blk**2, axis = 1)

#             S_crop[:,crop_i[0]:crop_i[1]] = stk_blk
        
#         print("progress: 100.00 %\n")
            
#         # add to std (finalise std)
#         std[S] = np.sqrt(std[S]/nwind)

#         # add to mean of I
#         if S == "I": # finalise mean of I
#             meanI += np.sum(S_crop, axis = 1)
#             meanI /= nwind
        
#         # normalize Stokes crop
#         S_crop -= meanI[:, None]
#         S_crop /= std[S][:, None]

#         # save stokes crop to file
#         np.save(f"{fname}_{S}.npy", S_crop)
#         print(f"Saved Stokes {S} dynamic spectra to: {fname}_{S}.npy")

#     print("removing temporary files...")
#     # remove temp files
#     os.remove("temp_Xfft.npy")
#     os.remove("temp_Yfft.npy")

#     print("Completed!")
#     print(f"Execution Time: {time() - t1:.2f} s")

#     return










    