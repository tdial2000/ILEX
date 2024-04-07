from ilex.frb import FRB
from ilex.data import *
from ilex.plot import *
import matplotlib.pyplot as plt
import numpy as np


frb = FRB(yaml_file = "scripts/211127.yaml")


frb.plot_crop(verbose = True)
