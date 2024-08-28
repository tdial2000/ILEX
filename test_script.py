from ilex.frb import FRB
from ilex.data import *
from ilex.plot import _PLOT
import matplotlib.pyplot as plt
from ilex.fitting import make_scatt_pulse_profile_func
import numpy as np
import sys

from ilex.logging import break_str, log_title, set_verbose


frb = FRB(yaml_file = "/fred/oz002/tdial/HTR_paper_data/230708/230708_updated.yaml", name = "Testing plots")
frb.set(verbose = False)

t = {}
for i in [1, 10, 20, 50]:
    dat = frb.get_data("fI", get = True, tN = i, fN = 1, t_crop = [1734, 1734.15])
    t[f"{i}"], t[f"{i}err"] = dat['fI'].copy(), dat['fIerr'].copy()


fig, ax = plt.subplots(1, 1, figsize = (10,10))

for i in [1, 10, 20, 50]:
    _PLOT(dat['freq'], t[f"{i}"], t[f"{i}err"], ax = ax, label = f"tN = {i}")

plt.legend()
plt.show()


# frb.set(RM = frb.fitted_params['RM']['rm'].val)

# frb.plot_PA()
