# Aprilgrid

### Pure python version of aprilgrid

<img src="example/showcase.avif" width="800" alt="Slow down for show case.">

## Install from PyPI
```
pip install aprilgrid
```

## Usage
Some examples of usage can be seen in the example/main.py file.

```py
from aprilgrid import Detector

detector = Detector("t36h11")

detector.detect(img)
```
## Development
```sh
git clone https://github.com/powei-lin/aprilgrid.git
cd aprilgrid
pip install -e .
```

## TODO
- [ ] Clean up unused debug code.
- [ ] Add output for opencv calibrate camera.
- [ ] Provide pregenerated pdf and easy way to customize.
- [ ] Support all tag families.
- [ ] Accelerate.
- [ ] Robustness.

## Reference
https://github.com/ethz-asl/kalibr