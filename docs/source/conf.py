# Configuration file for the Sphinx documentation builder.

import os, sys
sys.path.insert(0, os.path.abspath("../.."))
#'numpy', 'matplotlib', 'yaml', 'scipy', 'mpl_toolkits', 'bilby', 'RMtools_1D',
autodoc_mock_imports = [ 'data', 'fitting', 'plot', 'logging',
                        'master_proc', 'utils', 'globals', 'par', 'frb', 'htr', 'multicomp_pol']

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ILEX'
copyright = '2024, Tyson Dial'
author = 'Tyson Dial'
release = '0.9.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon',
              'sphinx.ext.autosummary']

autosummary_generate = True
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme_options = {'style_nav_header_background':'red'}
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]
html_static_path = ['_static']