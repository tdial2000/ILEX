# Welcome to ILEX
A Python packages for analysing High Time Resolution FRB data, dynamic spectra and polarisation. 
What does ILEX stand for? `¯\_(ツ)_/¯`


## Documentation
Here is the documentation for [ILEX](https://tdial2000.github.io/ILEX/), which includes function definitions and worked examples.


## Installing ILEX
To install ILEX, first clone the 'main' branch of the repo:

`git clone https://github.com/tdial2000/ILEX.git`

If you haven't got a python Virtual enviroment, you can make one simply using the following command OR the Conda enviroment equivelent:

`python -m venv --system-site-packages <venv_filepath>/<venv_name>`

Once the git repo is cloned, cd into `ilex\` and run the following command OR the Conda equivelent:

`pip install .`

The installed ILEX package can be used with an interactive python kernel `ipython`. To use A jupyter notebook, the 
relevant jupyter packages will also need to be installed as they are not included in ILEX. 


## TO DO LIST
There is a ['Trello'](https://trello.com/b/tRblRgMl/ilex-frb) board for ILEX development which list items still left to do. The main 
items left are:

1. Standardise fitting functions. There is a 'func_fit' wrapper method that will need to be re-written to enable any function to 
be sampled using BILBY. Also, Scipys curve_fit function using least squares is planned to be added as an additonal method of fitting.

2. Build Statistic class that controls the priors, posteriors and has a handful of class methods to calculate statistics such as
Chi-squared.

3. Fix up plotting for plot_stokes() and plot_poincare().

4. Fix up HTR code, (use what was made for new CELEBI code).

5. Implement saving function for data.
