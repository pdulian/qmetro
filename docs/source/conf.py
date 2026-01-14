# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QMetro++'
copyright = '2025, Piotr Dulian'
author = 'Piotr Dulian'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions =  [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    # 'sphinx_simplepdf',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinxcontrib.bibtex'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def setup(app):
    app.add_css_file("custom.css")

# Enable numbering of all displayed equations
math_number_all = False

# Bibliography settings
bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'unsrt'   # order of appearance
bibtex_reference_style = 'label'  # citation style e.g. author-year, label, numeric etc.

autosummary_generate = False

numfig = True
numfig_format = {
    'figure': 'Fig. %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
}

autodoc_typehints = "description"
