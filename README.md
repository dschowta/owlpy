# OWLPY

An open source time series library for Python implementing the [Matrix Profile](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html).

The Matrix Profile is a time-series analysis tool that can be used for several tasks e.g. motif/discord discovery.

Original project started by [Juan](https://github.com/jbeleno).

## Installation

Clone this repo and run:
```python
python setup.py install 
```

## Test

To test your installation, in the source directory run:
```python
python test.py 
```

This runs the Matrix Profile on two samples from the UCR Coffee Dataset, and plots the discord subsequence between the two, as well as the Matrix Profile representation.


## Usage

To use OWLPY, simply import from the _core_ package, as in:
```python
from owlpy.core import *
import numpy as np 
A = np.random.rand(100)
B = np.random.rand(100)
MP, I = stamp(A,B,15)
```
