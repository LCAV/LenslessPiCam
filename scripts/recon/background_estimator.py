import os

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.multimodal import CLIPImageQualityAssessment

from lensless.recon.utils import create_process_network
from lensless.utils.dataset import HFDataset

# To toake care of the warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer:
    """Trainer class for training and validating the background estimator model."""

    def __init__(self, model, optimizer, criterion, device, config):
        """
        Initializes the Trainer.

        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer for updating model parameters.
            criterion: The loss function.
            device: The device to run the model on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.clip_iqa = CLIPImageQualityAssessment(model_name_or_path="clip_iqa").to(device)

    def calculate_metrics(self, outputs, targets):
        """
        Calculates MAE, MSE, PSNR, SSIM, and CLIP-IQA between outputs and targets.

        Args:
            outputs: Model outputs (tensor).
            targets: Ground truth targets (tensor).

        Returns:
            A dictionary containing MAE, MSE, PSNR, SSIM, and CLIP-IQA.
        """
        # Ensure outputs and targets are in [0, 1]
        outputs = torch.clamp(outputs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        # If images have 1 channel, repeat to make 3 channels
        if outputs.shape[1] == 1:
            outputs = outputs.repeat(1, 3, 1, 1)
        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        # Resize images to 224x224 (CLIP input size)
        outputs_resized = F.interpolate(
            outputs, size=(224, 224), mode="bilinear", align_corners=False
        )
        targets_resized = F.interpolate(
            targets, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Compute CLIP-IQA
        with torch.no_grad():
            outputs_3d = outputs_resized
            if outputs_resized.shape[1] == 4:
                outputs_3d = outputs[:, :3, :, :]
            clip_iqa_score = self.clip_iqa(outputs_3d)  # , targets_resized)

        # Compute other metrics
        mae = F.l1_loss(outputs, targets).item()
        mse = F.mse_loss(outputs, targets).item()
        psnr = peak_signal_noise_ratio(outputs, targets).item()
        ssim = structural_similarity_index_measure(outputs, targets).item()

        return {
            "mae": mae,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "clip_iqa": clip_iqa_score.item(),
        }

    def calculate_metrics_batch(self, outputs, targets):
        """
        Calculates MAE, MSE, PSNR, SSIM, and CLIP-IQA between outputs and targets for batched data.

        Args:
            outputs: Model outputs (tensor) of shape [B, C, H, W].
            targets: Ground truth targets (tensor) of shape [B, C, H, W].

        Returns:
            A dictionary containing MAE, MSE, PSNR, SSIM, and CLIP-IQA.
        """
        # Ensure outputs and targets are in [0, 1]
        outputs = torch.clamp(outputs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        # If images have only one channel, repeat to make 3 channels
        if outputs.shape[1] == 1:
            outputs = outputs.repeat(1, 3, 1, 1)
        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        # Compute MAE and MSE over the batch
        mae = F.l1_loss(outputs, targets, reduction="mean").item()
        mse = F.mse_loss(outputs, targets, reduction="mean").item()

        # Compute PSNR and SSIM over the batch
        psnr = peak_signal_noise_ratio(outputs, targets, data_range=1.0).item()
        ssim = structural_similarity_index_measure(outputs, targets, data_range=1.0).item()

        # Compute CLIP-IQA
        with torch.no_grad():
            # Resize images to 224x224 for CLIP-IQA
            outputs_resized = F.interpolate(
                outputs, size=(224, 224), mode="bilinear", align_corners=False
            )

            outputs_3d = outputs_resized
            if outputs_resized.shape[1] == 4:
                outputs_3d = outputs[:, :3, :, :]
            clip_iqa_scores = self.clip_iqa(outputs_3d)

            # Compute CLIP-IQA scores over the batch
            clip_iqa = clip_iqa_scores.mean().item()

        return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim, "clip_iqa": clip_iqa}

    def crop_and_pad(self, batch):
        """
        Preprocesses a single sample by permuting, adding channels, and padding.

        Args:
            batch: A single data sample from the dataset.

        Returns:
            inputs: The preprocessed input.
            background: The preprocessed background.
        """

        inputs = batch[0][0].to(self.device)
        background = batch[2][0].to(self.device)

        # Permute inputs and background to (B, C, H, W)
        inputs = inputs.permute(0, 3, 1, 2)
        background = background.permute(0, 3, 1, 2)

        # Add a zero channel to inputs and background
        zeros_channel = torch.zeros(
            inputs.size(0), 1, inputs.size(2), inputs.size(3), device=self.device
        )

        inputs = torch.cat([inputs, zeros_channel], dim=1)

        background = torch.cat([background, zeros_channel], dim=1)

        # Compute desired height and width (multiples of 16)
        _, _, height, width = inputs.shape

        divisor = 16
        new_height = ((height - 1) // divisor + 1) * divisor
        new_width = ((width - 1) // divisor + 1) * divisor

        pad_height = new_height - height
        pad_width = new_width - width

        # Pad inputs and background
        inputs = F.pad(inputs, (0, pad_width, 0, pad_height), mode="reflect")
        background = F.pad(background, (0, pad_width, 0, pad_height), mode="reflect")

        return inputs, background

    def crop_and_pad_batch(self, data):
        """
        Preprocesses inputs and backgrounds by adding a zero channel and padding.

        Args:
            data: A batch from the DataLoader.
                data[0]: Inputs tensor of shape [B, 1, H_in, W_in, C]
                data[2]: Backgrounds tensor of shape [B, 1, H_bg, W_bg, C]

        Returns:
            inputs: Preprocessed inputs of shape [B, C+1, H_new, W_new]
            backgrounds: Preprocessed backgrounds of shape [B, C+1, H_new, W_new]
        """
        # Extract inputs and backgrounds
        inputs = data[0]  # Shape: [B, 1, H_in, W_in, C]
        backgrounds = data[2]  # Shape: [B, 1, H_bg, W_bg, C]

        # Remove the second dimension (of size 1)
        inputs = inputs.squeeze(1)  # Shape: [B, H_in, W_in, C]
        backgrounds = backgrounds.squeeze(1)  # Shape: [B, H_bg, W_bg, C]

        # Move tensors to the device
        inputs = inputs.to(self.device)
        backgrounds = backgrounds.to(self.device)

        # Permute to [B, C, H, W]
        inputs = inputs.permute(0, 3, 1, 2)  # Now [B, C, H_in, W_in]
        backgrounds = backgrounds.permute(0, 3, 1, 2)  # Now [B, C, H_bg, W_bg]

        # Add a zero channel to inputs and backgrounds
        zeros_channel_in = torch.zeros(
            inputs.size(0), 1, inputs.size(2), inputs.size(3), device=self.device
        )
        zeros_channel_bg = torch.zeros(
            backgrounds.size(0), 1, backgrounds.size(2), backgrounds.size(3), device=self.device
        )

        inputs = torch.cat([inputs, zeros_channel_in], dim=1)  # Now [B, C+1, H_in, W_in]
        backgrounds = torch.cat([backgrounds, zeros_channel_bg], dim=1)  # Now [B, C+1, H_bg, W_bg]

        # Determine new height and width (max of inputs and backgrounds)
        height = max(inputs.size(2), backgrounds.size(2))
        width = max(inputs.size(3), backgrounds.size(3))

        # Adjust height and width to be multiples of 16
        divisor = 16
        new_height = ((height - 1) // divisor + 1) * divisor
        new_width = ((width - 1) // divisor + 1) * divisor

        # Calculate padding for inputs
        pad_height_in = new_height - inputs.size(2)
        pad_width_in = new_width - inputs.size(3)

        # Calculate padding for backgrounds
        pad_height_bg = new_height - backgrounds.size(2)
        pad_width_bg = new_width - backgrounds.size(3)

        # Pad inputs and backgrounds
        inputs = F.pad(inputs, (0, pad_width_in, 0, pad_height_in), mode="reflect")
        backgrounds = F.pad(backgrounds, (0, pad_width_bg, 0, pad_height_bg), mode="reflect")

        return inputs, backgrounds

    def train(self, train_set, val_set, num_epochs, save_path="model.pth"):
        """
        Trains the model.

        Args:
            train_set: The training dataset.
            val_set: The validation dataset.
            num_epochs: Number of epochs to train.
            save_path: Path to save the trained model.
        """
        for epoch in range(num_epochs):
            self.model.train()

            running_loss = 0.0
            running_metrics = {"mae": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0, "clip_iqa": 0.0}

            # Training loop over the training dataset
            for data in train_set:
                data_processed = self.crop_and_pad_batch(data)
                inputs = data_processed[0].to(self.device)
                targets = data_processed[1].to(self.device)

                outputs = self.model(inputs)
                zeros_channel = torch.zeros(
                    outputs.size(0), 1, outputs.size(2), outputs.size(3), device=self.device
                )
                outputs = torch.cat([outputs, zeros_channel], dim=1)

                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                metrics = self.calculate_metrics_batch(outputs, targets)

                batch_size = inputs.size(0)
                for key in running_metrics.keys():
                    running_metrics[key] += metrics[key] * batch_size

            # Compute average metrics
            num_samples = len(train_set.dataset)
            avg_metrics = {k: v / num_samples for k, v in running_metrics.items()}

            avg_train_loss = running_loss / len(train_set)
            avg_mae = avg_metrics["mae"]
            avg_psnr = avg_metrics["psnr"]
            avg_ssim = avg_metrics["ssim"]
            # avg_clip_iqa = avg_metrics["clip_iqa"]

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"MAE: {avg_metrics['mae']:.4f}, "
                f"MSE: {avg_metrics['mse']:.4f}, "
                f"PSNR: {avg_metrics['psnr']:.2f}, "
                f"SSIM: {avg_metrics['ssim']:.4f}, "
                f"CLIP-IQA: {avg_metrics['clip_iqa']:.4f}"
            )

            # Validation phase
            avg_val_loss, val_metrics = self.validate(val_set)

            if self.config.wandb_project is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "train_mae": avg_mae,
                        "train_psnr": avg_psnr,
                        "train_ssim": avg_ssim,
                        "val_loss": avg_val_loss,
                        "val_mae": val_metrics["mae"],
                        "val_psnr": val_metrics["psnr"],
                        "val_ssim": val_metrics["ssim"],
                        "val_clip_iqa": val_metrics["clip_iqa"],
                    }
                )

                self.log_predictions(inputs, outputs, targets, epoch + 1)

        # Save the trained model
        self.save_model(save_path)

    def validate(self, val_set):
        """
        Validates the model on the validation set.

        Args:
            val_set: The validation dataset.

        Returns:
            avg_val_loss: Average validation loss.
        """
        self.model.eval()

        val_loss = 0.0
        val_metrics = {"mae": 0.0, "psnr": 0.0, "ssim": 0.0, "clip_iqa": 0.0}

        with torch.no_grad():
            for i, sample in enumerate(val_set):
                inputs, background = self.crop_and_pad_batch(sample)

                outputs = self.model(inputs)
                zeros_channel = torch.zeros(
                    outputs.size(0), 1, outputs.size(2), outputs.size(3), device=self.device
                )
                outputs = torch.cat([outputs, zeros_channel], dim=1)

                # Ensure outputs and background have the same shape
                loss = self.criterion(outputs, background)
                val_loss += loss.item()

                # Calculate metrics
                metrics = self.calculate_metrics_batch(outputs, background)
                val_metrics["mae"] += metrics["mae"]
                val_metrics["psnr"] += metrics["psnr"]
                val_metrics["ssim"] += metrics["ssim"]
                val_metrics["clip_iqa"] += metrics["clip_iqa"]

        avg_val_loss = val_loss / len(val_set)
        avg_mae = val_metrics["mae"] / len(val_set)
        avg_psnr = val_metrics["psnr"] / len(val_set)
        avg_ssim = val_metrics["ssim"] / len(val_set)
        avg_clip_iqa = val_metrics["clip_iqa"] / len(val_set)

        print(
            f"Validation Loss: {avg_val_loss:.4f}, MAE: {avg_mae:.4f}, "
            f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}"
        )

        return avg_val_loss, {
            "mae": avg_mae,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "clip_iqa": avg_clip_iqa,
        }

    def save_model(self, save_path="model.pth"):
        """
        Saves the model and optimizer state.

        Args:
            save_path: Path to save the model.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )

        if self.config.wandb_project is not None:
            wandb.save(save_path)
        print(f"Model and optimizer state saved to {save_path}")

    def log_predictions(self, inputs, outputs, background, epoch):
        """
        Logs predictions to WandB.

        Args:
            inputs: Input images.
            outputs: Model outputs.
            background: Ground truth backgrounds.
            epoch: Current epoch.
        """
        # Convert tensors to numpy arrays
        inputs_np = inputs.cpu().numpy()
        outputs_np = outputs.cpu().detach().numpy()
        background_np = background.cpu().numpy()

        # Remove batch dimension
        inputs_np = inputs_np.squeeze(0)  # Shape: [C, H, W]
        outputs_np = outputs_np.squeeze(0)  # Shape: [C, H, W]
        background_np = background_np.squeeze(0)  # Shape: [C, H, W]

        # Remove the extra channel from inputs
        if inputs_np.shape[0] > 3:
            inputs_np = inputs_np[:3, :, :]

        if outputs_np.shape[0] > 3:
            outputs_np = outputs_np[:3, :, :]

        if background_np.shape[0] > 3:
            background_np = background_np[:3, :, :]

        # Log images to WandB
        wandb.log(
            {
                "input": wandb.Image(inputs_np.transpose(1, 2, 0), caption=f"Input Epoch {epoch}"),
                "output": wandb.Image(
                    outputs_np.transpose(1, 2, 0), caption=f"Output Epoch {epoch}"
                ),
                "background": wandb.Image(
                    background_np.transpose(1, 2, 0), caption=f"Background Epoch {epoch}"
                ),
                "epoch": epoch,
            }
        )


def train_background_estimator(config, train_set, val_set):
    """
    Sets up the model, optimizer, and trainer, and starts the training process.

    Args:
        config: Configuration object from Hydra.
        train_set: The training dataset.
        val_set: The validation dataset.
    """
    # Create the background estimator model
    background_estimator, _ = create_process_network(
        config.reconstruction.pre_process.network,
        config.reconstruction.pre_process.depth,
        nc=config.reconstruction.pre_process.nc,
        device=config.torch_device,
        device_ids=config.device_ids,
        background_subtraction=config.reconstruction.integrated_background_unetres,
        input_background=config.reconstruction.unetres_input_background,
    )

    optimizer = optim.Adam(background_estimator.parameters(), lr=config.training.lr)
    criterion = nn.MSELoss()

    trainer = Trainer(
        background_estimator, optimizer, criterion, config.torch_device, config=config
    )

    trainer.train(train_set, val_set, config.training.num_epochs)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="train_background_estimator"
)
def main(config):
    """
    Main function to set up the dataset and start training.
    """

    # Load datasets
    print("Main")
    split_train = "train"
    split_test = "test"
    if config.files.split_seed is not None:
        from datasets import load_dataset, concatenate_datasets

        seed = config.files.split_seed
        generator = torch.Generator().manual_seed(seed)

        # Combine train and test into a single dataset
        train_split = "train"
        test_split = "test"
        if config.files.n_files is not None:
            train_split = f"train[:{config.files.n_files}]"
            test_split = f"test[:{config.files.n_files}]"
        print("Loading dataset")
        train_dataset = load_dataset(
            config.files.dataset, split=train_split, cache_dir=config.files.cache_dir
        )
        test_dataset = load_dataset(
            config.files.dataset, split=test_split, cache_dir=config.files.cache_dir
        )
        dataset = concatenate_datasets([test_dataset, train_dataset])

        # Split into train and test
        train_size = int((1 - config.files.test_size) * len(dataset))
        test_size = len(dataset) - train_size
        split_train, split_test = torch.utils.data.random_split(
            dataset, [train_size, test_size], generator=generator
        )

    print("Loading dataset object")
    train_set = HFDataset(
        huggingface_repo=config.files.dataset,
        cache_dir=config.files.cache_dir,
        psf=config.files.huggingface_psf,
        single_channel_psf=config.files.single_channel_psf,
        split=split_train,
        display_res=config.files.image_res,
        rotate=config.files.rotate,
        flipud=config.files.flipud,
        flip_lensed=config.files.flip_lensed,
        downsample=config.files.downsample,
        downsample_lensed=config.files.downsample_lensed,
        alignment=config.alignment,
        save_psf=config.files.save_psf,
        n_files=config.files.n_files,
        simulation_config=config.simulation,
        force_rgb=config.files.force_rgb,
        simulate_lensless=config.files.simulate_lensless,
        random_flip=config.files.random_flip,
        per_pixel_color_shift=config.files.per_pixel_color_shift,
        per_pixel_color_shift_range=config.files.per_pixel_color_shift_range,
        bg_snr_range=config.files.background_snr_range,
        bg_fp=(
            to_absolute_path(config.files.background_fp)
            if config.files.background_fp is not None
            else None
        ),
    )

    val_set = HFDataset(
        huggingface_repo=config.files.dataset,
        cache_dir=config.files.cache_dir,
        psf=config.files.huggingface_psf,
        single_channel_psf=config.files.single_channel_psf,
        split=split_test,
        display_res=config.files.image_res,
        rotate=config.files.rotate,
        flipud=config.files.flipud,
        flip_lensed=config.files.flip_lensed,
        downsample=config.files.downsample,
        downsample_lensed=config.files.downsample_lensed,
        alignment=config.alignment,
        save_psf=config.files.save_psf,
        n_files=config.files.n_files,
        simulation_config=config.simulation,
        per_pixel_color_shift=config.files.per_pixel_color_shift,
        per_pixel_color_shift_range=config.files.per_pixel_color_shift_range,
        bg_snr_range=config.files.background_snr_range,
        bg_fp=(
            to_absolute_path(config.files.background_fp)
            if config.files.background_fp is not None
            else None
        ),
        force_rgb=config.files.force_rgb,
        simulate_lensless=False,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if config.wandb_project is not None:
        wandb.init(project=config.wandb_project)

    # Start training
    print("Training background estimator")
    train_background_estimator(config, train_loader, val_loader)


if __name__ == "__main__":
    main()
