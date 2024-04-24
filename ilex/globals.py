# imports
from matplotlib.pyplot import Line2D
import inspect

class globals_:
  """
  Container for Global parameters

  Attributes
  ----------
  hkeys : List
    List of HTR crop products
  DATA : List
    List of HTR products
  p : Dict
    Dictionary of FRB parameters
  mp : Dict
    Dictionary of FRB meta-parameters
  hp : Dict
    Dictionary of FRB hyper-parameters
  
  """
  pass


_G = globals_()

#dict of HTR crop products
_G.hkeys = ["dsI","tI","fI",
            "dsQ","tQ","fQ",
            "dsU","tU","fU",
            "dsV","tV","fV",
            "X","Y"]


_G.DATA = "IQUVXY"


#dict of FRB parameters
_G.p = {"name": "FRBXXXXXX", "RA": "00:00:00.0000", "DEC": "00:00:00.0000",
                      "DM": 0.0, "bw": 336, "cfreq": 1271.5, "t_lim": [0.0, 3100.0],
                        "f_lim": [0.0,336.0], "nchan": 336, "nsamp": 0, "dt": 1e-3,
                          "df": 1.0, "RM":None, "f0":None, "pa0":0.0, "czap": ""}

# dict of frb meta params
_G.mp = {"t_crop": [0.0, 1.0], "f_crop":[0.0, 1.0], "terr_crop":None,
         "tN": 1, "fN": 1, "norm": "None"}

# dict of hyper parameters``
_G.hp = {'verbose': False, 'force': False, 'savefig': False, 'plot_type': 'scatter',
         'residuals': True, 'apply_tW': True, 'apply_fW': True}


_G.scatter_args = ['s','c','marker','alpha', 'cmap', 'vmin','vmax','linewidths','edgecolors', 'label']
_G.plot_args = []#list(inspect.signature(Line2D.set).parameters.keys())[1:]
_G.errorbar_args = ['fmt', 'ecolor', 'elinewidth', 'capsize', 'capthick', 'barsabove'] 
_G.errorbar_args += _G.plot_args



## constants
c = 2.997924538e8       # Speed of light [m/s]
kDM = 4.14938e3         # DM constant