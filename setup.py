import setuptools

# import version from file -> `__version__`
__version__ = None
with open("lensless/version.py") as f:
    exec(f.read())
assert __version__ is not None

# with open("README.rst", "r", encoding="utf-8") as fh:
#     long_description = fh.read()
long_description = "See the documentation at https://lensless.readthedocs.io/en/latest/"

setuptools.setup(
    name="lensless",
    version=__version__,
    author="Eric Bezzam",
    author_email="ebezzam@gmail.com",
    description="All-in-one package for lensless imaging: design, simulation, measurement, reconstruction.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/LCAV/LenslessPiCam",
    packages=setuptools.find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.1, <3.12",
    install_requires=[
        "opencv-python>=4.5.1.48",
        "numpy==1.26.4; python_version=='3.11'",
        "numpy>=1.22",
        "scipy>=1.7.0",
        "image>=1.5.33",
        "matplotlib>=3.4.2",
        "rawpy>=0.16.0",  # less than python 3.12
        "paramiko>=3.2.0",
        "hydra-core",
    ],
    extra_requires={"dev": ["pudb", "black"]},
)
