# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'quantmetrics'
copyright = '2025, Ella Elazkany'
author = 'Ella Elazkany'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.mathjax', 'sphinx.ext.autodoc','sphinx.ext.autosummary',
    'sphinx.ext.viewcode']

autosummary_generate = True

# Add the path to your Python source code
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))  # Adjust the path as needed

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


suppress_warnings = ['toc.not_included', 'ref.footnote']