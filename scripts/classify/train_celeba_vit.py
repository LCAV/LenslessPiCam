"""
Fine-tune ViT on CelebA dataset measured with lensless camera.
Original tutorial: https://huggingface.co/blog/fine-tune-vit

First, set-up HuggingFace libraries:
```
pip install datasets transformers[torch] scikit-learn tensorboardX
```

Raw measurement datasets can be download from SwitchDrive.
This will be done by the script if the dataset is not found.
```
# 10K measurements (13.1 GB)
python scripts/classify/train_celeba_vit.py \
data.measured=data/celeba_adafruit_random_2mm_20230720_10K

# 1K measurements (1.2 GB)
python scripts/classify/train_celeba_vit.py \
data.measured=data/celeba_adafruit_random_2mm_20230720_1K
```

Note that the CelebA dataset also needs to be available locally!
It can be download here: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

In order to classify on reconstructed outputs, the following
script needs to be run to create the dataset of reconstructed
images:
```
# reconstruct with ADMM
python scripts/recon/dataset.py algo=admm \
input.raw_data=path/to/raw/data
```

To classify on raw downsampled images, the same script can be
used, e.g. with the following command (`algo=null` for no reconstruction):
```
python scripts/recon/dataset.py algo=null \
input.raw_data=path/to/raw/data \
preprocess.data_dim=[48,64]
```

Other hyperparameters for classification can be found in
`configs/train_celeba_classifier.yaml`.

# TODO: update with Hugging Face dataset: https://huggingface.co/datasets/bezzam/DigiCam-CelebA-10K

"""

import warnings
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
import numpy as np
import torch
import os
from hydra.utils import to_absolute_path
import glob
import hydra
import random
from datasets import load_metric
from PIL import Image
import pandas as pd
import time
import torchvision.transforms as transforms
import torchvision.datasets as dset
from datasets import Dataset
from copy import deepcopy
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy

    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy

    def on_train_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


@hydra.main(version_base=None, config_path="../../configs", config_name="train_celeba_classifier")
def train_celeba_classifier(config):

    seed = config.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # check how many measured files
    measured_dataset = to_absolute_path(config.data.measured)
    if not os.path.isdir(measured_dataset):
        print(f"No dataset found at {measured_dataset}")
        try:
            from torchvision.datasets.utils import download_and_extract_archive
        except ImportError:
            exit()
        msg = "Do you want to download the CelebA dataset measured with a random Adafruit LCD pattern (13.1 GB)?"

        # default to yes if no input is given
        valid = input("%s (Y/n) " % msg).lower() != "n"
        if valid:
            url = "https://drive.switch.ch/index.php/s/9NNGCJs3DoBDGlY/download"
            filename = "celeba_adafruit_random_2mm_20230720_10K.zip"
            download_and_extract_archive(
                url, os.path.dirname(measured_dataset), filename=filename, remove_finished=True
            )
    measured_files = sorted(glob.glob(os.path.join(measured_dataset, "*.png")))
    print(f"Found {len(measured_files)} files in {measured_dataset}")

    if config.data.n_files is not None:
        n_files = config.data.n_files
        measured_files = measured_files[: config.data.n_files]
        print(f"Using {len(measured_files)} files")
    n_files = len(measured_files)

    # create dataset split
    attr = config.data.attr
    ds = dset.CelebA(
        root=config.data.original,
        split="all",
        download=False,
        transform=transforms.ToTensor(),
    )
    label_idx = ds.attr_names.index(attr)
    labels = ds.attr[:, label_idx][:n_files]

    # make dataset with measured data and corresponding labels
    df = pd.DataFrame(
        {
            "labels": labels,
            "image_file_path": measured_files,
        }
    )
    ds = Dataset.from_pandas(df)
    ds = ds.class_encode_column("labels")

    # -- train / test split
    test_size = config.data.test_size
    ds = ds.train_test_split(
        test_size=test_size, stratify_by_column="labels", seed=seed, shuffle=True
    )

    # prepare dataset
    model_name_or_path = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    # -- processors for train and val
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    # _train_transforms = Compose(
    #         [
    #             # RandomResizedCrop(
    #             #     size,
    #             #     scale=(0.9, 1.0),
    #             #     ratio=(0.9, 1.1),
    #             # ),
    #             Resize(size),
    #             CenterCrop(size),
    #             RandomHorizontalFlip(),
    #             ToTensor(),
    #             normalize,
    #         ]
    #     )
    _train_transforms = []
    if config.augmentation.random_resize_crop:
        _train_transforms.append(
            RandomResizedCrop(
                size,
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
            )
        )
    _train_transforms += [Resize(size), CenterCrop(size)]
    if config.augmentation.horizontal_flip:
        if config.data.raw:
            warnings.warn("Horizontal flip is not supported for raw data, Skipping!")
        else:
            _train_transforms.append(RandomHorizontalFlip())
    _train_transforms += [ToTensor(), normalize]
    _train_transforms = Compose(_train_transforms)

    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(examples):
        # Take a list of PIL images and turn them to pixel values
        examples["pixel_values"] = [
            _train_transforms(Image.open(fp)) for fp in examples["image_file_path"]
        ]
        return examples

    def val_transforms(examples):
        # Take a list of PIL images and turn them to pixel values
        examples["pixel_values"] = [
            _val_transforms(Image.open(fp)) for fp in examples["image_file_path"]
        ]
        return examples

    # transform dataset
    ds["train"].set_transform(train_transforms)
    ds["test"].set_transform(val_transforms)

    # data collator
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }

    # evaluation metric
    metric = load_metric("accuracy")

    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    # load model
    if config.train.prev is not None:
        model_path = to_absolute_path(config.train.prev)
    else:
        model_path = model_name_or_path

    labels = ds["train"].features["labels"].names
    model = ViTForImageClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        hidden_dropout_prob=config.train.dropout,
        attention_probs_dropout_prob=config.train.dropout,
    )

    # configure training
    output_dir = (
        config.data.output_dir + f"-{config.data.attr}" + os.path.basename(measured_dataset)
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.train.batch_size,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        num_train_epochs=config.train.n_epochs,
        fp16=True,
        logging_steps=10,
        learning_rate=config.train.learning_rate,
        save_total_limit=2,
        remove_unused_columns=False,  # important to keep False
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
    )
    trainer.add_callback(CustomCallback(trainer))  # add accuracy on train set

    # train
    hydra_output = os.getcwd()
    print("Results saved to : ", hydra_output)

    start_time = time.time()
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # evaluate
    metrics = trainer.evaluate(ds["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    metrics = trainer.evaluate(ds["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    #
    hydra_output = os.getcwd()
    print("Results saved to : ", hydra_output)
    print(f"Training took {time.time() - start_time} seconds")


if __name__ == "__main__":
    train_celeba_classifier()
