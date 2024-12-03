# imports
from matplotlib.pyplot import Line2D, rcParams
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
            "tL", "fL", "tP",
            "fP"]


_G.DATA = "IQUVXY"

# parameters that are changed during any instance call (i.e. **kwargs) that otherwise shouldn't be changed but may due to chaching
sp = ["t_crop", "terr_crop", "f_crop"]

_G.sp = {}
for item in sp:
  _G.sp[item] = f"{item}_static"

#dict of FRB parameters
_G.p = {"name": "FRBXXXXXX", "RA": "00:00:00.0000", "DEC": "00:00:00.0000", "MJD": 0.0,
                      "DM": 0.0, "bw": 336, "cfreq": 1271.5, "t_lim_base": [0.0, 3100.0],
                        "f_lim_base": [0.0,336.0], "nchan": 336, "nsamp": 0, "dt": 1e-3,
                          "df": 1.0, "RM":None, "f0":None, "pa0":0.0, "t_ref": 0.0}

# dict of frb meta params
_G.mp = {"t_crop": ["min", "max"], "f_crop":["min", "max"], "terr_crop":None,
         "tN": 1, "fN": 1, "norm": "None", "zapchan": None}

# dict of hyper parameters``
_G.hp = {'verbose': False, 'plot_type': 'lines',
         'residuals': True, 'apply_tW': True, 'apply_fW': True, 'zap': False, 'show_plots': True, 'save_plots': False, 'crop_units': "physical",
         'dynspec_cmap':'viridis'}

_G.yaml_ignore = ["zap"]

_G.yaml_def_prec = 8
_G.yaml_def_width = 3

# dict of keywords in yaml config file
_G.yaml_config = ['data', 'par', 'metapar', 'hyperpar', 'plots', 'fits', 'weights', 'multi']

_G.scatter_args = ['s','c','marker','alpha', 'cmap', 'vmin','vmax','linewidths','edgecolors', 'label']
# this command fails when building docs, so just brute forcing for now
#list(inspect.signature(Line2D.set).parameters.keys())[1:]
_G.plot_args = ['agg_filter',
 'alpha',
 'animated',
 'antialiased',
 'clip_box',
 'clip_on',
 'clip_path',
 'color',
 'dash_capstyle',
 'dash_joinstyle',
 'dashes',
 'data',
 'drawstyle',
 'fillstyle',
 'gapcolor',
 'gid',
 'in_layout',
 'label',
 'linestyle',
 'linewidth',
 'marker',
 'markeredgecolor',
 'markeredgewidth',
 'markerfacecolor',
 'markerfacecoloralt',
 'markersize',
 'markevery',
 'mouseover',
 'path_effects',
 'picker',
 'pickradius',
 'rasterized',
 'sketch_params',
 'snap',
 'solid_capstyle',
 'solid_joinstyle',
 'transform',
 'url',
 'visible',
 'xdata',
 'ydata',
 'zorder']
 
_G.errorbar_args = ['fmt', 'ecolor', 'elinewidth', 'capsize', 'capthick', 'barsabove'] 
_G.errorbar_args += _G.plot_args


# constants
_G.default_colors = rcParams['axes.prop_cycle'].by_key()['color']
_G.stk_colors = {"I":'k', "Q":_G.default_colors[1], "U":_G.default_colors[2], "V":'b', "L": 'r', "P": 'darkviolet'}


## constants
c = 2.997924538e8       # Speed of light [m/s]
kDM = 4.14938e3         # DM constant