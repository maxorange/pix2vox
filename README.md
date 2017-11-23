# pix2vox
[[Demonstration video]](https://maxorange.github.io/pix2vox/)<br>
Sketch-Based 3D Exploration with Stacked Generative Adversarial Networks.

<img src="img/sample.gif" width="500">

## Generated samples

### Single-category generation

<img src="img/single-category-generation.png" width="500">

### Multi-category generation

<img src="img/multi-category-generation.png" width="500">

## Requirements

The following python packages are required for running the application. If you are using [anaconda](https://www.continuum.io/), you can easily install VTK5 and PyQt4 (or they may already be installed).

* [binvox-rw-py](https://github.com/dimatura/binvox-rw-py)
* [numpy](https://github.com/numpy/numpy)
* [scipy](https://github.com/scipy/scipy)
* [OpenCV](http://opencv.org/)
* [TensorFlow](https://github.com/tensorflow/tensorflow) (GPU support is recommended).

* [VTK5](http://www.vtk.org/)

```
$ conda install -c anaconda vtk=5.10.1
```

* [PyQt4](https://www.riverbankcomputing.com/software/pyqt/intro)

```
$ conda install -c anaconda pyqt=4.11.4
```

* [QDarkStyleSheet](https://github.com/ColinDuquesnoy/QDarkStyleSheet)

```
$ pip install qdarkstyle
```

## Getting started

1. Install the python packages above.
2. Download the code from GitHub:

```
$ git clone https://github.com/maxorange/pix2vox.git
$ cd pix2vox
```

3. Run the code:

```
$ python application.py
```
