# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
import os
import sys
import importlib.metadata
dirname = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dirname, "..", __package__))
__version__ = importlib.metadata.version(__package__)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = __package__
author = 'tonegas'
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

templates_path = []
exclude_patterns = ['Thumbs.db', '.DS_Store','docs',
                    'nnodely/logger.py','nnodely/utils.py','nnodely/visualizer/*.*','nnodely/exporter/*.*',
                    'nnodely/initializer.py','nnodely/optimizer.py','nnodely/relation.py',
                    'nnodely.activation.py','nnodely.arithmetic.py','nnodely/fir.py','nnodely/fuzzify.py',
                    'nnodely/inpu.py','nnodely/linear.py','nnodely/localmodel.py','nnodely/output.py',
                    'nnodely/parametricfunction.py','nnodely/part.py','nnodely/parameter.py',
                    'nnodely/trigonometric.py']

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- Options for EPUB output -------------------------------------------------
epub_copyright = '2024, tonegas'  # Add this line
