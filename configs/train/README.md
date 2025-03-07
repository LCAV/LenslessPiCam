# Training configurations

The commands below train with the corresponding dataset, which will be downloaded from Hugging Face:

- [DiffuserCam](#diffusercam)
- [TapeCam](#tapecam)
- [DigiCam with a Single Mask](#digicam-with-a-single-mask)
- [DigiCam with Multiple Masks](#digicam-with-multiple-masks)
- [DigiCam CelebA](#digicam-celeba)
- [MultiPSF under External Illumination](#multipsf-under-external-illumination)

By commenting/uncommenting relevant sections in the configuration file, you can train models with different architectures or by setting parameters via the command line. 

By default, the model architecture uses five unrolleed iterations of ADMM for camera inversion, and UNetRes models for the pre-processor post-processor, and PSF correction.

With DiffuserCam, we show how to set different camera inversion methods.

## DiffuserCam

Dataset link: https://huggingface.co/datasets/bezzam/DiffuserCam-Lensless-Mirflickr-Dataset-NORM

The commands below show how to train different camera inversion methods on the DiffuserCam dataset (downsampled by a factor of 2 along each dimension). For a fair comparison, all models use around 8.1M parameters.

```bash
# unrolled ADMM
python scripts/recon/train_learning_based.py -cn diffusercam

# Trainable inversion (FlatNet but with out adversarial loss)
# -- need to set PSF as trainable
python scripts/recon/train_learning_based.py -cn diffusercam \
    reconstruction.method=trainable_inv \
    reconstruction.psf_network=False \
    trainable_mask.mask_type=TrainablePSF \
	trainable_mask.L1_strength=False

# Unrolled ADMM with compensation branch
# - adjust shapes of pre and post processors
python scripts/recon/train_learning_based.py -cn diffusercam \
    reconstruction.psf_network=False \
    reconstruction.pre_process.nc=[16,32,64,128] \
    reconstruction.post_process.nc=[16,32,64,128] \
    reconstruction.compensation=[24,64,128,256,400]

# Multi wiener deconvolution network (MWDN) 
# with PSF correction built into the network
python scripts/recon/train_learning_based.py -cn diffusercam \
    reconstruction.method=multi_wiener \
    reconstruction.multi_wiener.nc=[32,64,128,256,436] \
    reconstruction.pre_process.network=null \
    reconstruction.post_process.network=null \
    reconstruction.psf_network=False
```

### Multi PSF camera inversion

Similar to [PhoCoLens](https://phocolens.github.io/), we can train a camera inversion model that learns multiple PSFs. The training below uses the DiffuserCam dataset with its full resolution, and the number of model parameters is around 11.6M.
```bash
python scripts/recon/train_learning_based.py -cn diffusercam \
    reconstruction.method=svdeconvnet \
    reconstruction.pre_process.nc=[32,64,116,128] \
    reconstruction.psf_network=False \
    trainable_mask.mask_type=TrainablePSF \
	trainable_mask.L1_strength=False \
    files.downsample=1 files.downsample_lensed=1
```

## TapeCam

Dataset link: https://huggingface.co/datasets/bezzam/TapeCam-Mirflickr-25K

```bash
# unrolled ADMM
python scripts/recon/train_learning_based.py -cn tapecam
```

## DigiCam with a Single Mask

Dataset link: https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-SingleMask-25K

```bash
# unrolled ADMM
python scripts/recon/train_learning_based.py -cn digicam
```

## DigiCam with Multiple Masks

Dataset link: https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-25K

```bash
# unrolled ADMM
python scripts/recon/train_learning_based.py -cn digicam_multimask
```

## DigiCam CelebA

Dataset link: https://huggingface.co/datasets/bezzam/DigiCam-CelebA-26K

```bash
# unrolled ADMM
python scripts/recon/train_learning_based.py -cn digicam_celeba
```

## MultiPSF under External Illumination

Dataset link: https://huggingface.co/datasets/Lensless/MultiLens-Mirflickr-Ambient

```bash
# unrolled ADMM
python scripts/recon/train_learning_based.py -cn multilens_ambient
```
