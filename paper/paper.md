---
title: 'LenslessPiCam: A hardware and software toolkit for lensless imaging with a Raspberry Pi'
tags:
  - lensless imaging
  - inverse problems
  - Raspberry Pi
  - Python
  - computational imaging
authors:
  - name: Eric Bezzam
    orcid: 0000-0003-4837-5031 
    affiliation: 1
  - name: Martin Vetterli
    orcid: 0000-0002-6122-1216
    affiliation: 1
  - name: Matthieu Simeoni
    orcid: 0000-0002-4927-3697
    affiliation: 1
affiliations:
 - name: École Polytechnique Fédérale de Lausanne (EPFL)
   index: 1
date: 4 March 2022
bibliography: paper.bib
   
---

# Summary

Lensless imaging seeks to replace/remove the lens in a 
conventional imaging setup. The earliest cameras were in fact 
lensless, relying on 
long exposure times to form images on the other end of a small aperture in a 
darkened room/container (*camera obscura*). The introduction of a lens
allowed for more light throughput and therefore shorter exposure 
times, while retaining sharp focus. The 
incorporation of digital sensors
readily enabled computational techniques to post-process and enhance images.
Lensless imaging make this post-processing a part of the imaging mechanism, 
thereby removing the need to form a viewable image at the sensor, e.g. with a lens. 
It represents a paradigm shift in camera system design as there is more flexibility to cater 
the hardware to the application at hand (e.g. lightweight or flat). As a 
consequence,
the imaging software is typically faced with solving an inverse problem in order
to recover an image of the underlying object. 
For a comprehensive, theoretical treatment of lensless imaging, we refer to 
[@boominathan2022recent]. 
With `LenslessPiCam`, we provide an accessible hardware and software 
toolkit to enable researchers, hobbyists, and students to implement and 
explore practical aspects of lensless imaging.


# Statement of need

Being at the interface of hardware, software, and algorithm design, the field of
lensless imaging necessitates a broad array of competances that might deter 
newcomers to the field. The purpose of `LenslessPiCam` is to provide a
complete toolkit with cheap, 
accessible hardware designs and open-source software, all the while achieving 
satisfactory results in order to explore novel ideas for hardware, software, and
algorithm design.

The DiffuserCam tutorial [@diffusercam] served as a great starting point when
developing our toolkit, as it demonstrates that a working lensless camera can be
built with cheap hardware: a Raspberry Pi, the Camera Module 2,[^1] and a piece 
of tape. The
authors also provide Python implementations of two inverse problem reconstruction 
approaches: variants of gradient descent (GD) with a non-negativity contraint 
*and* alternating direction method of multipliers (ADMM) [@boyd2011distributed]
with a non-negativity constraint and a total variation (TV) regularizer. Moreover,
detailed guides explain how to build their camera and give intuition behind the 
reconstruction algorithms. \autoref{fig:compare_cams}a shows a reconstruction we
obtained after replicating the DiffuserCam tutorial. 

[^1]: [www.raspberrypi.com/products/camera-module-v2](https://www.raspberrypi.com/products/camera-module-v2).

Reproducing the tutorial gave us considerable insight into the 
practical challenges of developing a lensless camera. However, as can be seen 
from \autoref{fig:compare_cams}a, the resolution is very poor and the tutorial is limited to
grayscale reconstruction. With `LenslessPiCam`, we improve the resolution by
using the newer HQ camera[^2] and extend the application to RGB imaging. An 
example reconstruction can be seen in \autoref{fig:compare_cams}b. Furthermore, we
incorporate Pycsou [@pycsou], a modular Python package for solving linear inverse
problems, in order to provide flexibility in choosing penalties/regularizers.

[^2]: [www.raspberrypi.com/products/raspberry-pi-high-quality-camera/](https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/).

![Alternating direction method of multipliers (ADMM) reconstruction of thumbs-up on a phone 40 cm away.](compare_cams.png){#fig:compare_cams}

<!-- | DiffuserCam   | `LenslessPiCam`   |
|-------------  |---------------  |
| ![](hq_cam.png){#fig:compare_cams width=45%} | ![](hq_cam.png){#fig:compare_cams width=45%} | -->


`LenslessPiCam` is designed to be used by researchers, hobbyists, and students.
In the past, we have found such open-source hardware and software platforms to be a valuable 
resource for researchers [@bezzam2017hardware] and students alike [@bezzam2019teaching].

# Contributions

With respect to the DiffuserCam tutorial [@diffusercam], we have made the following
contributions.

In terms of hardware, as shown in \autoref{fig:hardware}, we:

- make use of the HQ camera sensor ($50): 4056 x 3040 pixels (12.3 MP) and 7.9 mm 
sensor diagonal, compared to 3280 × 2464 pixels (8.1 MP) and 4.6 mm sensor diagonal for the 
Camera Module 2 ($30). 
- provide the design and firmware for a cheap point source generator (needed 
for calibration), which consists of an Arduino, a white LED, and a cardboard box.
  
![(a) LenslessPiCam, (b) point source generator (inside), and (c) point source generator (outside).](hardware.png){#fig:hardware}


With respect to reconstruction algorithms, we:

- provide significantly faster implementations of GD and ADMM, i.e. around 3x 
  reduction in computation time.
- extend the above reconstructions to RGB.
- provide an object-oriented structure that is easy to extend for exploring new 
  algorithms.
- provide an object-oriented interface to Pycsou for solving
  lensless imaging inverse problems. Pycsou is a Python package
  for solving linear inverse problems of the form
\begin{equation}\label{eq:fourier}
\min_{\mathbf{\alpha}\in\mathbb{R}^N} \,F(\mathbf{y}, \mathbf{G} \mathbf{\alpha})\quad+\quad \lambda\mathcal{R}(\mathbf{\alpha}),
\end{equation}
where $F$ is a data-fidelity term between the observed and predicted 
measurements $\mathbf{y}$ and $\mathbf{G}\mathbf{\alpha}$ respectively, 
$\mathcal{R}$ is a regularization component (could consist of more than one prior), 
and $\lambda >0$ controls the amount of regularization,

We also provide functionality to:

- remotely capture Bayer data with the proposed camera.
- convert Bayer data to RGB or grayscale.
- quantitavely evaluate the point spread function (PSF) of the lensless camera.
- remotely display data on an external monitor, which can be used to automate 
  raw data measurements to, e.g., gather a dataset.
- evalute reconstructions on a variety of metrics: MSE, PSNR, SSIM, LPIPS [@zhang2018perceptual].

Finally, we have written a set Medium articles to guide the process of building
and using the proposed lensless camera. An overview of these articles can be found [here](https://medium.com/@bezzam/a-complete-lensless-imaging-tutorial-hardware-software-and-algorithms-8873fa81a660).
The articles also include a set of proposed exercises for students.

# API

The core algorithmic component of `LenslessPiCam` is the abstract class 
[`lensless.recon.ReconstructionAlgorithm`](https://github.com/LCAV/DiffuserCam/blob/70936c1a1d0797b50190d978f8ece3edc7413650/diffcam/recon.py#L9).
The three reconstruction strategies available in `LenslessPiCam` 
derive from this class:

- `lensless.gradient_descent.GradientDescient`: projected gradient descent 
  with a non-negativity constraint. Two accelerated approaches are also
  available: `lensless.gradient_descent.NesterovGradientDescent` 
  [@nesterov1983method] and `lensless.gradient_descent.FISTA` [@beck2009fast].
- `lensless.admm.ADMM`: alternating direction method of multipliers (ADMM) with
  a non-negativity constraint and a total variation (TV) regularizer.
- `lensless.apgd.APGD`: accelerated proximal gradient descent with Pycsou
as a backend. Any differentiable or proximal operator can be used as long as it 
  is compatible with Pycsou, namely derives from one of 
  `DifferentiableFunctional` or `ProximableFunctional` from Pycsou.
  
New reconstruction algorithms can be conveniently implemented by deriving from 
the abstract class and defining the following abstract methods:

- the update step: `_update`.
- a method to reset state variables: `reset`.
- an image formation method: `_form_image`. 
  
One advantage of deriving from `lensless.recon.ReconstructionAlgorithm` is that
functionality for iterating, saving, and visualization is already implemented. 
Consequently, using a reconstruction algorithm that derives from it boils down 
to three steps:

1. Creating an instance of the reconstruction algorithm.
2. Setting the data.
3. Applying the algorithm.

For example, for ADMM (full example in `scripts/admm.py`):
```python
    recon = ADMM(psf)
    recon.set_data(data)
    res = recon.apply(n_iter=n_iter)
```

A template for applying a reconstruction algorithm (including loading the data)
can be found in `scripts/reconstruction_template.py`.


# Efficient reconstruction

In the table below, we compare the processing time of DiffuserCam's and 
`LenslessPiCam`'s implementations for grayscale reconstruction of:

1. gradient descent using FISTA and a non-negativity constraint;
2. ADMM with a non-negativity constraint and a TV regularizer.

The DiffuserCam implementations can be found 
[here](https://github.com/Waller-Lab/DiffuserCam-Tutorial), while 
`lensless.apgd.APGD` and `lensless.admm.ADMM` are used for `LenslessPiCam`. The 
comparison is done on a Lenovo Thinkpad P15 running Ubuntu 21.04.

[comment]: <> (|               |   GD   |  APGD  |  APGD &#40;real&#41;  |  ADMM  |)

[comment]: <> (|:-------------:|:------:|:------:|:------:|:------:|)

[comment]: <> (|  DiffuserCam  |  215 s | - | - | 7.24 s |)

[comment]: <> (| `LenslessPiCam` | 70.1 s | 93.2 s | 67.9 s | 2.76 s |)

[comment]: <> (|               |   GD   |  GD &#40;real&#41;  |  ADMM  |)

[comment]: <> (|:-------------:|:------:|:------:|:------:|)

[comment]: <> (|  DiffuserCam  |  215 s | - | 7.24 s |)

[comment]: <> (| `LenslessPiCam` | 93.2 s | 67.9 s | 2.76 s |)

[comment]: <> (: Benchmark grayscale reconstruction. 300 iterations for gradient descent &#40;GD&#41;)

[comment]: <> (and 5 iterations for alternating direction method of multipliers &#40;ADMM&#41;. )

[comment]: <> (`lensless.apgd.APGD` is used in the case of `LenslessPiCam` and GD. For GD )

[comment]: <> (&#40;real&#41;, the convolution in the Fourier domain exploits the fact that we are )

[comment]: <> (dealing with real 2D signals.)

|               |   GD   |  ADMM  |
|:-------------:|:------:|:------:|
|  DiffuserCam  |  215 s | 7.24 s |
| `LenslessPiCam` | 67.9 s | 2.76 s |
: Benchmark grayscale reconstruction. 300 iterations for gradient descent (GD)
and 5 iterations for alternating direction method of multipliers (ADMM).

From the above table, we observe a 3.1x reduction in computation time for
GD and a 2.6x reduction for ADMM. This comes from:

- our object-oriented implementation of the algorithms, which allocates all the 
  necessary memory beforehand and pre-computes everything data-independent, such
  as forward operators from the point spread function (PSF).
- our use of the real FFT, which is possible since we are working with
real-valued image intensities.

\autoref{fig:grayscale} shows the corresponding grayscale reconstruction for 
FISTA and ADMM, which are equivalent for both DiffuserCam and `LenslessPiCam`.

![Grayscale reconstruction using FISTA (a) and ADMM (b).](grayscale.png){#fig:grayscale}

# Quantifying performance

In order to compare different reconstruction approaches, it is necessary to
quantify the performance. To this end, `LenslessPiCam` provides functionality
to extract regions of interest from the reconstruction and compare it with the
original image via multiple metrics:

- Mean-squared error (MSE).
- Peak signal-to-noise ratio (PSNR).
- Mean structural similarity (SSIM) index.
- Learned perceptual image patch similarity (LPIPS).

Below is an example of how a reconstruction can be evaluated against an original
image.[^3]

[^3]: Using `scripts/compute_metrics_from_original.py`.

![Extracting region from \autoref{fig:compare_cams}b to quantify performance.](metric.png){#fig:metric width=90%}

|  MSE  | PSNR |  SSIM | LPIPS |
|:-----:|------|:-----:|:-----:|
| 0.164 | 7.85 | 0.405 | 0.645 |
: Metrics for \autoref{fig:metric}.

Sometimes it may be of interest to perform an exhaustive evaluation on a large
dataset.
While `LenslessPiCam` could be used for collecting such a dataset with the
proposed camera,[^4] the 
authors of [@monakhova2019learned] have already collected a dataset of 25'000 
parallel measurements, namely 25'000 pairs of DiffuserCam and lensed camera images.[^5]
`LenslessPiCam` offers functionality to evaluate a reconstruction algorithm on
this dataset, or a subset of it that we have prepared.[^6] Note that this 
dataset is collected with a different lensless camera, but is nonetheless of 
significant value for exploring reconstruction techniques.

[^4]: Using the remote display and capture scripts, i.e. 
`scripts/remote_display.py` and `scripts/remote_capture.py` respectively.

[^5]: [waller-lab.github.io/LenslessLearning/dataset.html](https://waller-lab.github.io/LenslessLearning/dataset.html).

[^6]: Subset of [DiffuserCam Lensless Mirflickr Dataset](https://waller-lab.github.io/LenslessLearning/dataset.html)
consists of 200 files (725 MB) as opposed to 25'000 files (100 GB) of the
original dataset. The subset can be downloaded [here](https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE).

Table 3 shows the average metric results after applying 100 iterations of ADMM
to the subset we have prepared.[^7]

[^7]: Using `scripts/evaluate_mirflickr_admm.py`.

|  MSE  | PSNR |  SSIM | LPIPS |
|:-----:|------|:-----:|:-----:|
| 0.0797 | 12.7 | 0.535 | 0.585 |
: Average metrics for subset (200 files) of the DiffuserCam Lensless Mirflickr Dataset.

One can also visualize the performance on a single file of the dataset, namely
how the reconstruction changes as the number of iterations increase.[^8] The 
final reconstruction and outputed metrics are shown in 
\autoref{fig:dataset_single_file} and Table 4.

[^8]: Using `scripts/apply_admm_single_mirflickr.py`.

![Visualizing performance of ADMM (100 iterations) on a single file of the DiffuserCam Lensless Mirflickr Dataset.](dataset_single_file.png){#fig:dataset_single_file}

|  MSE  | PSNR |  SSIM | LPIPS |
|:-----:|------|:-----:|:-----:|
| 0.0682 | 11.7 | 0.486| 0.504 |
: Metrics for \autoref{fig:dataset_single_file}.

# As an educational resource

As mentioned earlier, `LenslessPiCam` can serve as an educational 
resource. We have used it in our graduate-level signal processing source for
providing experience in applying fundamental signal processing concepts and 
solving linear inverse problems. The work of our students can be found 
[here](https://infoscience.epfl.ch/search?ln=en&rm=&ln=en&sf=&so=d&rg=10&c=Infoscience%2FArticle&c=Infoscience%2FBook&c=Infoscience%2FChapter&c=Infoscience%2FConference&c=Infoscience%2FDataset&c=Infoscience%2FLectures&c=Infoscience%2FPatent&c=Infoscience%2FPhysical%20objects&c=Infoscience%2FPoster&c=Infoscience%2FPresentation&c=Infoscience%2FProceedings&c=Infoscience%2FReport&c=Infoscience%2FReview&c=Infoscience%2FStandard&c=Infoscience%2FStudent&c=Infoscience%2FThesis&c=Infoscience%2FWorking%20papers&c=Media&c=Other%20doctypes&c=Work%20done%20outside%20EPFL&c=&of=hb&fct__2=LCAV&p=diffusercam).

As exercises in implementing key signal processing components, we have left a 
couple incomplete functions in `LenslessPiCam`:

- `lensless.autocorr.autocorr2d`: to compute a 2D autocorrelation in the 
  frequency domain;
- `RealFFTConvolve2D`: to perform a convolution in the frequency domain with the 
  real FFT and vectorize operations for RGB.
  
We have also proposed a few reconstruction approaches to implement in
[this Medium article](https://medium.com/@bezzam/lensless-imaging-with-the-raspberry-pi-and-python-diffusercam-473e47662857).

For the solutions to the above implementations, please request access to 
[this folder](https://drive.google.com/drive/folders/1Y1scM8wVfjVAo5-8Nr2VfE4b6VHeDSia?usp=sharing).

# Conclusion

In summary, `LenslessPiCam` provides all the necessary hardware designs and 
software to build, use, and evaluate a lensless camera with cheap and accessible
components. As we continue to use it as a research and educational platform, we
hope to investigate and incorporate:

- computational refocusing.
- programmable masks.
- data-driven, machine learning reconstruction techniques.

# Acknowledgements

We acknowledge feedback from Sepand Kashani, Julien Fageot, and the students 
during the first iteration of this project in our graduate course.

# References