# Configuration file for the Sphinx documentation builder.
import datetime
from lensless import __version__


autodoc_mock_imports = ["numpy", "scipy", "cupy", "cupyx"]

# -- Project information

project = "LenslessPiCam"
copyright = f"{datetime.date.today().year}, Eric Bezzam"
author = "Eric Bezzam"
version = __version__
release = version

# -- General configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "NumPy [latest]": ("https://docs.scipy.org/doc/numpy/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("http://matplotlib.sourceforge.net/", None),
    "pycsou": ("https://matthieumeo.github.io/pycsou/html/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
# html_theme_options = {"navigation_depth": -1, "titles_only": True}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
