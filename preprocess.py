#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 11:27:20 2022

@author: gjadick

For functions that preprocess the projection data.
"""

import numpy as np


def get_G(gamma_coord, cols, s, fc):
    ''' get an FT-domain fanbeam ramp-like filter '''
    freq_real = np.fft.rfftfreq(cols, d=s)
    H = np.abs(freq_real)
    H[H > fc] = 0.0
    gamma_FT = np.fft.rfft((gamma_coord/np.sin(gamma_coord))**2)  # RFFT of gamma func
    G = np.convolve(gamma_FT, H, mode='full')[:len(H)]  # positive freqs, 0 at 0
    return G


def get_w3D(a, a_m, kl):
    # get initial weighting based on tan function
    w3D_raw = 1 / (1 + (np.tan(np.abs(a))/np.tan(np.abs(a_m)))**kl)
    # rescale weights to range 0 to 1 (instead of 0.5 to 1)
    w3D = 2*(w3D_raw - 0.5) + 1-np.max(w3D_raw)
    return w3D
