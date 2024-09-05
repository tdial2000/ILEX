# Welcome to ILEX
A Python packages for analysing High Time Resolution FRB data, dynamic spectra and polarisation. 


## Documentation
Here is the documentation for [ILEX](https://tdial2000.github.io/ILEX/), which includes function definitions and worked examples.


## Installing ILEX
To install ILEX, first clone the repo:

`git clone https://github.com/tdial2000/ILEX.git -b <branch>`

If you haven't got a python Virtual enviroment, you can make one simply using the following command OR the Conda enviroment equivelent:

`python -m venv --system-site-packages <venv_filepath>/<venv_name>`

Once the git repo is cloned, cd into the folder and run the following command in the case you are using a venv:

`source install_ilex.sh`

If using a Conda enviroment and the above script is not used, you will need to set the `ILEX_PATH`enviromental variable in the 
`ilex/__init__.py` file.

The installed ILEX package can be used with an interactive python kernel `ipython`. To use A jupyter notebook, the 
relevant jupyter packages will also need to be installed as they are not included in ILEX. 


## TO DO LIST
There is a ['Trello'](https://trello.com/b/tRblRgMl/ilex-frb) board for ILEX development which list items still left to do.

## NOTE
ILEX is a pokemon reference.
