# imports
from ilex.frb import FRB
from ilex.logging import strcol


# create new FRB instance with VELA data
# NOTE: Even though this is mainly an FRB toolkit, any radio data with full stokes polarisation S(f,t) can 
# be used.
frb = FRB("../examples/VELA240621.yaml")

try:
    # lets first find VELA
    frb.find_frb(method = "fluence", mode = "min", rms_guard = 5, rms_width = 10, 
                    rms_offset = 20)

    print(frb.metapar.t_crop)
    print("[FINDING FRB] " + strcol("PASS", 'green'))
except:
    print("[FINDING FRB] " + strcol("FAIL", 'red'))


# set some parameters
frb.set(show_plots = False, save_plots = True, verbose = False)

# zap freq channels at zero
try:
    # zap
    frb.zap_channels(zapzeros = True)

    # also zap anything above 3500 MHz
    frb.zap_channels(chans = "3500:4500")

    print("[ZAPPING DATA] " + strcol("PASS", 'green'))

except:
    print("[ZAPPING DATA] " + strcol("FAIL", 'red'))


# plot stokes freq parameters
try:
    # scunch a few bins around centroid (zero point)
    frb.plot_stokes(stk_type = "f", stk2plot = "ILVP", stk_ratio = True)

    print("[PLOTTING STK DATA] " + strcol("PASS", 'green'))

except: 
    print("[PLOTTING STK DATA] " + strcol("FAIL", 'red'))