Tutorial 4: ILEX config files
------------------------------

Overview
========

For ease of use, users have the option of creating an FRB config file that holds all the parameters, metaparameters, hyperparameters as well
as plotting/fitting options that can easily be tweaked in one placed. These config files are also used in the 
additional ILEX scripts provided. A config file can be made by either using the ``make_config.py`` script or by directly copying the 
``defaut.yaml`` file in ``ilex/files/`` directory. An FRB config file looks something like the following:

.. code-block:: yaml

    data:  # file paths for stokes dynamic spectra
      dsI: "220610_dsI.npy"
      dsQ: "220610_dsQ.npy"
      dsU: "220610_dsU.npy"
      dsV: "220610_dsV.npy"

    par:   # parameters
      name:   "FRB220610"
      RA:     "00:00:00.0000"
      DEC:    "00:00:00.0000"
      DM:     0.0
      bw:     336
      MJD:    0
      cfreq:  1271.5
      t_lim_base:  [0.0, 3100.0]
      f_lim_base:  [0.0, 336.0]
      t_ref:  0
      nchan:  336
      nsamp:  0
      dt:     0.05
      df:     4.0
      RM:     null
      f0:     null
      pa0:    0.0

    metapar:   # metaparameters
      t_crop:     [20.9, 23.8]
      f_crop:     [1103.5, 1200]
      terr_crop:  null
      tN:         1
      fN:         1
      zapchan:    null
      norm:       "None"

    hyperpar:   # hyperparameters
      verbose: false
      show_plots: true
      save_plots: false
      plot_type: lines
      residuals: true
      plotPosterior: true
      apply_tW: true
      apply_fW: true
      crop_units: physical
      dynspec_cmap: arctic
      plot_tpad: 30.0
      show_dynzaps: true
      dynspec_satlvl: 0
      dynspec_cnorm: linear
      dynspec_cmap_alpha: 0.5
      mplstyle: /fred/oz002/tdial/ILEX/files/default.mplstyle
      zap: true



Using the FRB config file
=========================

An FRB config file can be used when either creating an FRB instance or when using the ``frb.load_data()`` method. All 
the FRB parameters will then be loaded in and ready to use!

.. code-block:: python

    # import
    from ilex.frb import FRB

    # create FRB instance with config file
    frb = FRB(yaml_file = "examples/220610.yaml") # or frb = FRB("examples/220610.yaml")


.. image:: 220610_dsI_crop.png
   :width: 720pt


Saving the FRB instance as a config file
========================================

The ``.save_data()`` method allows the user to save an FRB instance as a .yaml file for quick use later. To do so,
set ``save_yaml = True`` and provide a name for the yaml file using the ``yaml_file`` argument. If the FRB instance was 
created by loading in a yaml file in the first place, that file will be overwritten unless ``yaml_file`` is specified. If not, 
then the .yaml file name will be set to ``<FRB.name>.yaml``. 

.. code-block:: python

    # save frb instance as a .yaml file
    frb.save_data(save_yaml = True, yaml_file = "FRB220610.yaml")

