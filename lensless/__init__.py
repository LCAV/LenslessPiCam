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


from .recon.recon import ReconstructionAlgorithm
from .recon.admm import ADMM
from .recon.gd import (
    GradientDescent,
    NesterovGradientDescent,
    FISTA,
    GradientDescentUpdate,
)
from .recon.tikhonov import CodedApertureReconstruction
from .hardware.sensor import VirtualSensor, SensorOptions

try:
    from .recon.trainable_recon import TrainableReconstructionAlgorithm
    from .recon.unrolled_admm import UnrolledADMM
    from .recon.unrolled_fista import UnrolledFISTA
except Exception:
    pass

try:
    from .recon.apgd import APGD, APGDPriors

    pycsou_available = True
except Exception:
    pycsou_available = False

from .version import __version__
