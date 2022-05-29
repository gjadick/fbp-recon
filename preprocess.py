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


def ramp_filter(vec, H):
    ''' in FT domain, filter a 1D vector with ramp filter H '''
    FT = np.fft.rfft(vec)
    FT_filtered = np.fft.ifftshift(H*np.fft.fftshift(FT))
    return np.fft.irfft(FT_filtered)


def get_w3D(a, a_m, kl):
    return 1 / (1 + (np.tan(np.abs(a))/np.tan(np.abs(a_m)))**kl)
