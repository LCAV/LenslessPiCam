import setuptools

# import version from file -> `__version__`
__version__ = None
with open("lensless/version.py") as f:
    exec(f.read())
assert __version__ is not None

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lensless",
    version=__version__,
    author="Eric Bezzam",
    author_email="ebezzam@gmail.com",
    description="Package to control and image with a lensless camera running on a Raspberry Pi.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LCAV/LenslessPiCam",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.1",
    install_requires=[
        "opencv-python>=4.5.1.48",
        "numpy>=1.22, <=1.23.5",
        "scipy>=1.7.0",
        "image>=1.5.33",
        "matplotlib>=3.4.2",
        "rawpy>=0.16.0",
    ],
    extra_requires={"dev": ["pudb", "black"]},
)
