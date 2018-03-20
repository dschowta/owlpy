import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('../ECGFiveDays_TRAIN', sep=',', header=None)
label = data.pop(data.columns[0])

def plot_motif(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8,4))
    plt.subplot(211)
    plt.plot(Ta, linestyle='--', alpha=0.5)
    plt.xlim((0, len(Ta)))
    
    print(np.argmax(values))
    
    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g', label='Top Motif')
    plt.plot(range(np.argmax(values), np.argmax(values) + m), Ta[np.argmax(values):np.argmax(values) + m], c='r', label='Top Discord')
    
    plt.legend(loc='best')
    plt.title('Time-Series')


    plt.subplot(212)
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

print(data.head())

ts = data.iloc[:10].values.flatten() # pick a random sample from class 0

from owlpy.core import *

Pab, Iab = stamp(ts,ts,100)                       # run the STAMP algorithm to compute the Matrix Profile
plot_motif(ts,ts,Pab,Iab,100)