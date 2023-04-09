# Aprilgrid

aprilgrid: Pure python version of aprilgid

## Install from PyPI
This is the recommended and easiest way to install aprilgrid.
```
pip install aprilgrid
```
We offer pre-built binary wheels for common operating systems. To install from source, see below.

## Usage
Some examples of usage can be seen in the src/pupil_apriltags/bindings.py file.

The Detector class is a wrapper around the Apriltags functionality. You can initialize it as following:
```py
from aprilgrid import Detector

at_detector = Detector("t36h11")

at_detector.detect(img)
```
# clone the repository
git clone https://github.com/powei-lin/aprilgrid.git
cd aprilgrid

# install apriltags in editable mode with development requirements
```sh
pip install -e .[testing]
```