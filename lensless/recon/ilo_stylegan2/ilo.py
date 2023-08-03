"""
TODO : authorship from original ILO
https://github.com/giannisdaras/ilo/blob/master/ilo_stylegan.py
"""

import numpy as np
import math
from tqdm import tqdm
from PIL import Image
from hydra.utils import to_absolute_path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from . import (
    lpips as lpips,
)  # TODO : linking needs to be fixed, or use LPIPS from torchmetrics: https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html
from .stylegan2 import Generator
from .utils import project_onto_l1_ball, zero_padding_tensor
from waveprop.devices import sensor_dict, SensorParam
from lensless.recon.rfft_convolve import RealFFTConvolve2D


torch.set_printoptions(precision=5)
torch.autograd.set_detect_anomaly(True)


# def get_transformation(image_size):
#     return transforms.Compose(
#         [transforms.Resize(image_size),
#          transforms.ToTensor(),
#          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def get_transformation():
    return transforms.Compose([transforms.ToTensor()])


# Latent z -> latent w
class MappingProxy(nn.Module):
    def __init__(self, gaussian_ft):
        super(MappingProxy, self).__init__()
        self.mean = gaussian_ft["mean"]
        self.std = gaussian_ft["std"]
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu(self.std * x + self.mean)
        return x


def loss_geocross(latent):
    if latent.size()[1:] == (1, 512):
        return 0
    else:
        num_latents = latent.size()[1]
        X = latent.view(-1, 1, num_latents, 512)
        Y = latent.view(-1, num_latents, 1, 512)
        A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
        D = 2 * torch.atan2(A, B)
        D = ((D.pow(2) * 512).mean((1, 2)) / 8.0).mean()
        return D


class SphericalOptimizer:
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {
                param: (param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt()
                for param in params
            }

    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_(
                (param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt()
            )
            param.mul_(self.radii[param])


class LatentOptimizer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["opti_params"]["device"]

        # Load models and pre-trained weights
        gen = Generator(1024, 512, 8)
        gen.load_state_dict(
            torch.load(to_absolute_path(config["model"]["checkpoint"]))["g_ema"], strict=False
        )
        gen.eval()
        self.gen = gen.to(self.device)
        self.gen.start_layer = config["opti_params"]["start_layer"]
        self.gen.end_layer = config["opti_params"]["end_layer"]
        self.mpl = MappingProxy(
            torch.load(
                to_absolute_path("/scratch/ilo_lensless/checkpoint/gaussian_fit.pt"), self.device
            )
        )

        cuda_ids = [0]
        if self.device.startswith("cuda:"):
            cuda_ids = self.device.split(":")[-1].split(",")
            cuda_ids = [int(cuda_id) for cuda_id in cuda_ids]

        self.percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=self.device.startswith("cuda"), gpu_ids=cuda_ids
        )

        # Transform on each image
        self.transform = get_transformation()

        # Task
        image_size = np.array(config["preprocessing"]["resize"]["image_size"])

        # Load PSF
        psf_path = config["lensless_imaging"]["psf_path"]
        psf_image = np.array(Image.open(to_absolute_path(psf_path)).convert("RGB"))

        self.lensless_imaging = True
        scene2mask = config["lensless_imaging"]["scene2mask"]
        mask2sensor = config["lensless_imaging"]["mask2sensor"]
        object_height = config["lensless_imaging"]["object_height"]
        sensor_config = sensor_dict[config["lensless_imaging"]["sensor"]]
        self.psf_size = np.array(config["lensless_imaging"]["psf_size"])

        # Input image at the right size
        magnification = mask2sensor / scene2mask
        scene_dim = sensor_config[SensorParam.SIZE] / magnification
        object_height_pix = int(np.round(object_height / scene_dim[1] * self.psf_size[1]))
        scaling = object_height_pix / image_size[1]
        self.object_dim = (np.round(image_size * scaling)).astype(int).tolist()

        # Normalize and forward model
        psf_image = psf_image / np.sum(psf_image, axis=(0, 1))
        self.psf_image = (
            torch.from_numpy(psf_image).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        # Create forward model
        self.forward_model = RealFFTConvolve2D(self.psf_image)

        # Opti parameters
        self.start_layer = config["opti_params"]["start_layer"]
        self.end_layer = config["opti_params"]["end_layer"]
        self.steps = config["opti_params"]["steps"]

        self.lr = config["opti_params"]["lr"]
        self.lr_same_pace = config["opti_params"]["lr_same_pace"]

        self.project = config["opti_params"]["project"]
        self.do_project_latent = config["opti_params"]["do_project_latent"]
        self.do_project_noises = config["opti_params"]["do_project_noises"]
        self.do_project_gen_out = config["opti_params"]["do_project_gen_out"]

        self.max_radius_latent = config["opti_params"]["max_radius_latent"]
        self.max_radius_noises = config["opti_params"]["max_radius_noises"]
        self.max_radius_gen_out = config["opti_params"]["max_radius_gen_out"]

        # Loss parmaters
        self.geocross = config["loss_params"]["geocross"]
        self.mse = config["loss_params"]["mse"]
        self.pe = config["loss_params"]["pe"]
        self.dead_zone_linear = config["loss_params"]["dead_zone_linear"]
        self.dead_zone_linear_alpha = config["loss_params"]["dead_zone_linear_alpha"]
        self.lpips_method = config["loss_params"]["lpips_method"]

        # Logs parameters
        self.save_gif = config["logs"]["save_gif"]
        self.save_every = config["logs"]["save_every"]
        self.save_forward = config["logs"]["save_forward"]

    def init_state(self, input_files):

        # Initialize the state of the optimizer, has to be performed before every run

        self.layer_in = None
        self.best = None
        self.best_forward = None
        self.current_step = 0

        # Load images
        input_images = []
        for input_file in input_files:
            input_images.append(self.transform(Image.open(input_file).convert("RGB")))
        self.input_images = torch.stack(input_images, 0).to(self.device)
        self.batchsize = self.input_images.shape[0]

        # Initialization of latent vector
        noises_single = self.gen.make_noise(self.batchsize)
        self.noises = []
        for noise in noises_single:
            self.noises.append(noise.normal_())
        self.latent_z = torch.randn(
            (self.batchsize, 18, 512), dtype=torch.float, requires_grad=True, device=self.device
        )
        self.gen_outs = [None]

    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp

    def invert_(self, start_layer, noise_list, steps, index):
        learning_rate_init = self.lr[index]
        print(f"Running round {index + 1} / {len(self.steps)} of ILO.")

        # noise_list containts the indices of nodes that we will be optimizing over
        for i in range(len(self.noises)):
            if i in noise_list:
                self.noises[i].requires_grad = True
            else:
                self.noises[i].requires_grad = False
        with torch.no_grad():
            if start_layer == 0:
                var_list = [self.latent_z] + self.noises
            else:
                self.gen_outs[-1].requires_grad = True
                var_list = [self.latent_z] + self.noises + [self.gen_outs[-1]]
                prev_gen_out = (
                    torch.ones(self.gen_outs[-1].shape, device=self.gen_outs[-1].device)
                    * self.gen_outs[-1]
                )
            prev_latent = (
                torch.ones(self.latent_z.shape, device=self.latent_z.device) * self.latent_z
            )
            prev_noises = [
                torch.ones(noise.shape, device=noise.device) * noise for noise in self.noises
            ]

            # set network that we will be optimizing over
            self.gen.start_layer = start_layer
            self.gen.end_layer = self.end_layer

        # Optimizer
        optimizer = optim.Adam(var_list, lr=self.lr[index])
        ps = SphericalOptimizer([self.latent_z] + self.noises)
        pbar = tqdm(range(steps))
        self.current_step += steps

        # Loss
        mse_loss = 0
        p_loss = 0

        for i in pbar:
            # Update learning rate
            if self.lr_same_pace:
                total_steps = sum(self.steps)
                t = i / total_steps
            else:
                t = i / steps

            lr = self.get_lr(t, learning_rate_init)
            optimizer.param_groups[0]["lr"] = lr

            # Update generated image
            latent_w = self.mpl(self.latent_z)

            img_gen, _ = self.gen(
                [latent_w],
                input_is_latent=True,
                noise=self.noises,
                layer_in=self.gen_outs[-1],
            )

            # Normalize output of GAN from standardize to [0, 1] per batch
            img_gen = torch.clamp(img_gen, -1, 1)
            img_gen = 0.5 * img_gen + 0.5

            # Calculate loss
            loss = 0

            # TODO : check if image always generated on 1024x1024
            # Downsample to the original size
            A_img_gen = img_gen
            # A_img_gen = self.downsampler_1024_image(img_gen)
            # A_img_gen = F.interpolate(A_img_gen, size=, mode='bicubic')

            A_img_gen = F.interpolate(A_img_gen, size=self.object_dim, mode="bicubic")
            A_img_gen = zero_padding_tensor(A_img_gen, self.psf_size)
            A_img_gen = self.forward_model(A_img_gen)

            # Using the all range [0,1]
            with torch.no_grad():
                max_vals = torch.max(torch.flatten(A_img_gen, start_dim=1), dim=1)[0]
                max_vals = max_vals.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            A_img_gen /= max_vals

            # Calculate perceptual loss (LPIPS)
            if self.pe[index] != 0:
                if self.lpips_method == "default":
                    p_loss = self.percept(A_img_gen, self.input_images, normalize=True).mean()
                # elif self.lpips_method == 'fill_mask':
                #     # TODO : maybe need to downsampeld if super resolution
                #     filled = self.mask * self.input_images + (1 - self.mask) * img_gen
                #     p_loss = self.percept(self.downsampler_image_256(A_img_gen), self.downsampler_image_256(filled), normalize=True).mean()
                else:
                    raise NotImplementedError("LPIPS policy not implemented")
            loss += self.pe[index] * p_loss

            # Calculate dead_zone_linear loss
            diff = torch.abs(A_img_gen - self.input_images) - self.dead_zone_linear_alpha
            loss += (
                self.dead_zone_linear[index]
                * torch.max(torch.zeros(diff.shape, device=diff.device), diff).mean()
            )

            # Calculate MSE loss
            mse_loss = F.mse_loss(A_img_gen, self.input_images)
            loss += self.mse[index] * mse_loss

            # Calculate Geocross loss
            loss += self.geocross * loss_geocross(self.latent_z[:, start_layer:])

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Backpropagate on projections
            if self.project:
                ps.step()

            if self.max_radius_gen_out[index] == float("inf"):
                self.do_project_gen_out = False

            if self.max_radius_latent[index] == float("inf"):
                self.do_project_latent = False

            if self.max_radius_noises[index] == float("inf"):
                self.do_project_noises = False

            if start_layer != 0 and self.do_project_gen_out:
                deviation = project_onto_l1_ball(
                    self.gen_outs[-1] - prev_gen_out, self.max_radius_gen_out[index]
                )
                var_list[-1].data = (prev_gen_out + deviation).data
            if self.do_project_latent:
                deviation = project_onto_l1_ball(
                    self.latent_z - prev_latent, self.max_radius_latent[index]
                )
                var_list[0].data = (prev_latent + deviation).data
            if self.do_project_noises:
                deviations = [
                    project_onto_l1_ball(noise - prev_noise, self.max_radius_noises[index])
                    for noise, prev_noise in zip(self.noises, prev_noises)
                ]
                for i, deviation in enumerate(deviations):
                    var_list[i + 1].data = (prev_noises[i] + deviation).data

            # Update best image
            # if mse_loss < mse_min:
            #     mse_min = mse_loss
            #     self.best = img_gen

            #     if self.save_forward:
            #         self.best_forward = A_img_gen

            self.best = img_gen

            if self.save_forward:
                self.best_forward = A_img_gen

            # Update tqdm and print
            pbar.set_description((f"perceptual: {p_loss:.4f};" f" mse: {mse_loss:.4f};"))
            # TODO : probably broken coz of batch
            # Save some intermediate images of the optimization
            if self.save_gif and i % self.save_every == 0:
                torchvision.utils.save_image(
                    img_gen,
                    f"gif_{start_layer}_{i}.png",
                    nrow=int(img_gen.shape[0] ** 0.5),
                    normalize=True,
                )

        # Update in between layers
        with torch.no_grad():
            latent_w = self.mpl(self.latent_z)
            self.gen.end_layer = self.gen.start_layer
            intermediate_out, _ = self.gen(
                [latent_w],
                input_is_latent=True,
                noise=self.noises,
                layer_in=self.gen_outs[-1],
                skip=None,
            )
            self.gen_outs.append(intermediate_out)
            self.gen.end_layer = self.end_layer
        print()

    def invert(self):
        print("Start of the invertion")
        for i, steps in enumerate(self.steps):
            begin_from = i + self.start_layer
            if begin_from > self.end_layer:
                raise Exception("Attempting to go after end layer...")
            self.invert_(begin_from, range(5 + 2 * begin_from), int(steps), i)

        return (
            self.input_images,
            (self.latent_z, self.noises, self.gen_outs),
            (self.best, self.best_forward),
        )
