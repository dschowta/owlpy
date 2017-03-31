import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('../Coffee_TRAIN', sep=',', header=None)
label = data.pop(data.columns[0])
print(data.head())

d = (data[label==0].sample().values.flatten())
e = (data[label==1].sample().values.flatten())

from owlpy.core import *

Pdd=test_stamp(d,e,25)
