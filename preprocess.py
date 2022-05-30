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


def get_w3D(a, a_m, kl, detail_mode=False):
    '''
    Get weights for 3D imaging, to help correct some cone beam artifacts.
    Returns weights between 0.5 and 1 (or between 0 and 1 for "detail_mode")
    - a: "alpha" [rad] the cone beam angle of interest
    - a_m: "alpha_max" [rad] the maximum cone beam angle for the 2D detector array
    - kl: a correction weighting parameter between 0 and 1
    - detail_mode: (bool) rescale weights to range from 0 to 1
    '''
    # get initial weighting based on tan function
    w3D_raw = 1 / (1 + (np.tan(np.abs(a))/np.tan(np.abs(a_m)))**kl)
    
    # for detail mode, rescale weights to range 0 to 1 (instead of 0.5 to 1)
    if detail_mode:
        w3D = 2*(w3D_raw - 0.5) + 1-np.max(w3D_raw)
    
    return np.float32(w3D)
