from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from . import dist_model


class PerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        model="net-lin",
        net="alex",
        colorspace="rgb",
        spatial=False,
        use_gpu=True,
        gpu_ids=[0],
    ):
        super(PerceptualLoss, self).__init__()
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dist_model.DistModel()
        self.model.initialize(
            model=model,
            net=net,
            use_gpu=use_gpu,
            colorspace=colorspace,
            spatial=self.spatial,
            gpu_ids=gpu_ids,
        )

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        return self.model.forward(target, pred)
