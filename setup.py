import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diffcam",
    version="0.0.1",
    author="Eric Bezzam",
    author_email="ebezzam@gmail.com",
    description="Package to control and image with DiffuserCam running on a Raspberry Pi.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LCAV/DiffuserCam",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "opencv-python==4.5.1.48",
        "numpy==1.20.3",
        "scipy==1.7.0",
        "click==8.0.1",
        "image==1.5.33",
        "matplotlib==3.4.2",
        "jedi==0.18.0",
        "picamerax==20.9.1",
        "pycsou==1.0.6",
        "lpips==0.1.4",
        "rawpy==0.16.0",
        "scikit-image==0.19.0rc0",
    ],
    extra_requires={
        "dev": ["pudb", "black"],
    },
)
