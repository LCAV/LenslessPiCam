# #############################################################################
# tikhonov.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# Aaron FARGEON [aa.fargeon@gmail.com]
# #############################################################################

"""
Tikhonov
========

The py:class:`~lensless.recon.tikhonov.CodedApertureReconstruction` class is meant
to recover an image from a py:class:`~lensless.hardware.mask.CodedAperture` lensless
capture, using the analytical solution to the Tikhonov optimization problem
(least squares problem with L2 regularization term), as in the `FlatCam paper
<https://arxiv.org/abs/1509.00116>`_ (Eq. 7).
"""

import numpy as np
from numpy.linalg import multi_dot

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class CodedApertureReconstruction:
    """
    Reconstruction method for the (non-iterative) Tikhonov algorithm presented in the `FlatCam paper <https://arxiv.org/abs/1509.00116>`_.

    TODO: operations in float32
    """

    def __init__(self, mask, image_shape, P=None, Q=None, lmbd=3e-4):
        """
        Parameters
        ----------
        mask : py:class:`lensless.hardware.mask.CodedAperture`
            Coded aperture mask object.
        image_shape : (`array-like` or `tuple`)
            The shape of the image to reconstruct.
        P : :py:class:`~numpy.ndarray`, optional
            Left convolution matrix in measurement operator. Must be of shape (measurement_resolution[0], image_shape[0]).
            By default, it is generated from the mask. In practice, it may be useful to measure as in the FlatCam paper.
        Q : :py:class:`~numpy.ndarray`, optional
            Right convolution matrix in measurement operator. Must be of shape (measurement_resolution[1], image_shape[1]).
            By default, it is generated from the mask. In practice, it may be useful to measure as in the FlatCam paper.
        lmbd: float:
            Regularization parameter. Default value is `3e-4` as in the FlatCam paper `code <https://github.com/tanjasper/flatcam/blob/master/python/demo.py>`_.
        """

        self.lmbd = lmbd
        if P is None or Q is None:
            self.P, self.Q = mask.get_conv_matrices(image_shape)
        else:
            self.P = P
            self.Q = Q
        assert self.P.shape == (
            mask.resolution[0],
            image_shape[0],
        ), "Left matrix P shape mismatch"
        assert self.Q.shape == (
            mask.resolution[1],
            image_shape[1],
        ), "Right matrix Q shape mismatch"

    def apply(self, img):
        """
        Method for performing Tikhinov reconstruction.

        Parameters
        ----------
        img : :py:class:`~numpy.ndarray` or :py:class:`torch.Tensor`
            Lensless capture measurement. Must be 3D even if grayscale.

        Returns
        -------
        :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Reconstructed image, in the same format as the measurement.
        """
        assert (
            len(img.shape) == 3
        ), "Object should be a 3D array or tensor (HxWxC) even if grayscale."

        if torch_available and isinstance(img, torch.Tensor):

            # Empty matrix for reconstruction
            n_channels = img.shape[-1]
            x_est = torch.empty([self.P.shape[1], self.Q.shape[1], n_channels])

            self.P = torch.from_numpy(self.P).float()
            self.Q = torch.from_numpy(self.Q).float()

            # Applying reconstruction for each channel
            for c in range(n_channels):
                Yc = img[:, :, c]

                # SVD of left matrix
                UL, SL, VLh = torch.linalg.svd(self.P)
                VL = VLh.T
                DL = torch.cat(
                    (
                        torch.diag(SL),
                        torch.zeros([self.P.shape[0] - SL.size(0), SL.size(0)], device=SL.device),
                    )
                )
                singLsq = SL**2

                # SVD of right matrix
                UR, SR, VRh = torch.linalg.svd(self.Q)
                VR = VRh.T
                DR = torch.cat(
                    (
                        torch.diag(SR),
                        torch.zeros([self.Q.shape[0] - SR.size(0), SR.size(0)], device=SR.device),
                    )
                )
                singRsq = SR**2

                # Applying analytical reconstruction
                inner = torch.linalg.multi_dot([DL.T, UL.T, Yc, UR, DR]) / (
                    torch.outer(singLsq, singRsq) + torch.full(x_est.shape[0:2], self.lmbd)
                )
                x_est[:, :, c] = torch.linalg.multi_dot([VL, inner, VR.T])

            # Non-negativity constraint: setting all negative values to 0
            x_est = torch.clamp(x_est, min=0)

            # Normalizing the image
            x_est = (x_est - x_est.min()) / (x_est.max() - x_est.min())

        else:

            # Empty matrix for reconstruction
            n_channels = img.shape[-1]
            x_est = np.empty([self.P.shape[1], self.Q.shape[1], n_channels])

            # Applying reconstruction for each channel
            for c in range(n_channels):

                # SVD of left matrix
                UL, SL, VLh = np.linalg.svd(self.P, full_matrices=True)
                VL = VLh.T
                DL = np.concatenate((np.diag(SL), np.zeros([self.P.shape[0] - SL.size, SL.size])))
                singLsq = np.square(SL)

                # SVD of right matrix
                UR, SR, VRh = np.linalg.svd(self.Q, full_matrices=True)
                VR = VRh.T
                DR = np.concatenate((np.diag(SR), np.zeros([self.Q.shape[0] - SR.size, SR.size])))
                singRsq = np.square(SR)

                # Applying analytical reconstruction
                Yc = img[:, :, c]
                inner = multi_dot([DL.T, UL.T, Yc, UR, DR]) / (
                    np.outer(singLsq, singRsq) + np.full(x_est.shape[0:2], self.lmbd)
                )
                x_est[:, :, c] = multi_dot([VL, inner, VR.T])

            # Non-negativity constraint: setting all negative values to 0
            x_est = x_est.clip(min=0)

            # Normalizing the image
            x_est = (x_est - x_est.min()) / (x_est.max() - x_est.min())

        return x_est
