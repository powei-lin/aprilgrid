# Aprilgrid

aprilgrid: Pure python version of aprilgrid

## Install from PyPI
This is the recommended and easiest way to install aprilgrid.
```
pip install aprilgrid
```

## Usage
Some examples of usage can be seen in the example/main.py file.

```py
from aprilgrid import Detector

at_detector = Detector("t36h11")

at_detector.detect(img)
```
# clone the repository
```sh
git clone https://github.com/powei-lin/aprilgrid.git
cd aprilgrid
```
# install apriltags in editable mode with development requirements
```sh
pip install -e .[testing]
```