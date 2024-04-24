from ilex.frb import FRB
from ilex.data import *
from ilex.plot import _PLOT
import matplotlib.pyplot as plt
import numpy as np
import sys


x = np.linspace(2150, 2250, 250)

y = np.cos(x)

yerr = 0.05 * np.ones(250)

_PLOT(x = x, y = y, yerr = yerr, color = 'r', plot_type = "scatter")


plt.show()
# frb = FRB(yaml_file = "examples/220610.yaml")

# frb.plot_data("tI")
# frb.plot_stokes(stk_type = "t", stk_ratio = True, plot_type = "scatter", sigma = 10.0)



