# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys
import importlib.metadata
dirname = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dirname, "..", __package__))
__version__ = importlib.metadata.version(__package__)

# -- Project information -----------------------------------------------------

project = __package__
author = 'tonegas'
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = []
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- Options for EPUB output -------------------------------------------------
epub_copyright = '2024, tonegas'  # Add this line
