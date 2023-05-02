# #############################################################################
# __init__.py
# ===========
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Lensless library
===========================

Package for designing, simulating, and imaging with lensless cameras.
"""


from .recon import ReconstructionAlgorithm
from .admm import ADMM
from .gradient_descent import (
    GradientDescent,
    NesterovGradientDescent,
    FISTA,
    GradientDescentUpdate,
)
from .sensor import VirtualSensor, SensorOptions

try:
    from .apgd import APGD, APGDPriors

    pycsou_available = True
except Exception:
    pycsou_available = False

from .version import __version__
