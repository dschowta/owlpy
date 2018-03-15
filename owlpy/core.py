'''--------------------------- Core Script ---------------------------------'''
'''
    Description: This library is based on the algorithms described in 
    [1] Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, 
        Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, 
        Eamonn Keogh (2016). Matrix Profile I: All Pairs Similarity Joins 
        for Time Series: A Unifying View that Includes Motifs, Discords and 
        Shapelets. IEEE ICDM 2016.
'''
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt


def slidingDotProduct(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m:n]


def calculateDistanceProfile(q, t, qt, a, sum_q, sum_q2, mean_t, sigma_t):
    n = t.size
    m = q.size

    b = np.zeros(n - m)
    dist = np.zeros(n - m)
    for i in range(0, n - m):
        b[i] = -2 * (qt[i].real - sum_q * mean_t[i]) / sigma_t[i]
        dist[i] = a[i] + b[i] + sum_q2
    return np.sqrt(np.abs(dist))


# The code below takes O(m) for each subsequence
# you should replace it for MASS
def computeMeanStdForQuery(Q):
    # Compute Q stats -- O(n)
    sumQ = np.sum(Q)
    sumQ2 = np.sum(np.power(Q, 2))
    return sumQ, sumQ2


def preComputeMeanStdForTS(ta, m):
    na = len(ta)
    sum_t = np.zeros(na - m)
    sum_t2 = np.zeros(na - m)

    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    for i in range(na - m):
        sum_t[i] = cumulative_sum_t[i + m] - cumulative_sum_t[i]
        sum_t2[i] = cumulative_sum_t2[i + m] - cumulative_sum_t2[i]
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(sigma_t2)
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


# MUEEN’S ALGORITHM FOR SIMILARITY SEARCH (MASS)
def mass(Q, T, a, meanT, sigmaT):
    # Z-Normalisation
    if np.std(Q) != 0:
        Q = (Q - np.mean(Q)) / np.std(Q)
    QT = slidingDotProduct(Q, T)
    sumQ, sumQ2 = computeMeanStdForQuery(Q)
    return calculateDistanceProfile(Q, T, QT, a, sumQ, sumQ2, meanT, sigmaT)


def elementWiseMin(Pab, Iab, D, idx, ignore_trivial, m):
    for i in range(0, len(D)):
        if not ignore_trivial or (np.abs(idx - i) > m):
            if D[i] < Pab[i]:
                Pab[i] = D[i]
                Iab[i] = idx
    return Pab, Iab


def stamp(Ta, Tb, m):
    nb = len(Tb)
    na = len(Ta)
    Pab = np.full(nb - m, float('Inf'))
    Iab = np.full(nb - m, 0)
    idxes = range(0, nb - m)

    sumT, sumT2, meanT, meanT_2, meanTP2, sigmaT, sigmaT2 = preComputeMeanStdForTS(Ta, m)

    a = np.zeros(na - m)
    for i in range(0, na - m):
        a[i] = (sumT2[i] - 2 * sumT[i] * meanT[i] + m * meanTP2[i]) / sigmaT2[i]

    for idx in idxes:
        D = mass(Tb[idx: idx + m], Ta, a, meanT, sigmaT)
        Pab, Iab = elementWiseMin(Pab, Iab, D, idx, ignore_trivial=(Ta == Tb).all(), m=m)

    return Pab, Iab


# Quick Test
def test_stamp(Ta, Tb, m):
    start_time = time.time()

    Pab, Iab = stamp(Ta, Tb, m)
    print("--- %s seconds ---" % (time.time() - start_time))

    plot_graphics(Ta, Tb, Pab, Iab, m)
    return Pab


def plot_graphics(Ta, Tb, values, indexes, m):
    fig_width = 16
    fig_height = 8
    fig_dpi = 100
    plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

    plt.subplot(411)
    plt.plot(Ta)
    plt.xlim((0, len(Ta)))
    plt.title('A')

    plt.subplot(412)
    plt.plot(Tb)
    plt.plot(range(np.argmax(values), np.argmax(values) + m), Tb[np.argmax(values):np.argmax(values) + m], c='r')
    plt.title('B')
    plt.xlim((0, len(Tb)))

    plt.subplot(413)
    plt.title('P_ab')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', ms=10)
    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.subplot(414)
    plt.title('I_ab')
    plt.plot(range(0, len(indexes)), indexes, '#ff5722')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.xlim((0, len(Ta)))
    plt.show()



