#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 11:27:20 2022

@author: gjadick

For functions that preprocess the projection data.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

def get_H(p, cols, gauss_sigma=1):
    ''' get an FT-domain ramp filter '''
    freq = np.fft.rfftfreq(cols)
    N = freq.size
    H = np.array([np.abs(x) for x in np.linspace(-freq[N//2], freq[N//2], N)])   
    H[H > p*np.max(H)] = 0.0
    H = gaussian_filter1d(H, gauss_sigma)
    return H

def get_G(gamma_coord, cols, s, fc):
    ''' get an FT-domain fanbeam ramp-like filter '''
    freq_real = np.fft.rfftfreq(cols, d=s)
    H = np.abs(freq_real)
    H[H > fc] = 0.0
    gamma_FT = np.fft.rfft((gamma_coord/np.sin(gamma_coord))**2)  # RFFT of gamma func
    G = np.convolve(gamma_FT, H, mode='full')[:len(H)]  # positive freqs, 0 at 0
    return G


def ramp_filter(vec, H):
    ''' in FT domain, filter a 1D vector with ramp filter H '''
    FT = np.fft.rfft(vec)
    FT_filtered = np.fft.ifftshift(H*np.fft.fftshift(FT))
    return np.fft.irfft(FT_filtered)






def get_w3D(a, a_m, kl):
    return 1 / (1 + (np.tan(np.abs(a))/np.tan(np.abs(a_m)))**kl)
