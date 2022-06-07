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
    else:
        w3D = w3D_raw
    
    return np.float32(w3D)


def get_conjugate_proj(i_beta, q_orig, cols, gamma_coord, vz_coord, dbeta_proj, dz_proj):
    '''
    Gets the conjugate projection for projection q_orig[i_beta],
    using the fan-beam redundancy q(gamma, beta) = q(-gamma, beta + 2*gamma + pi)
    '''
    q_conjugate = np.zeros(q_orig[0].shape)
    gamma_max = np.max(gamma_coord)
    dgamma = gamma_coord[1]-gamma_coord[0]
    dvz = vz_coord[1]-vz_coord[0]

    for i_gamma in  range(cols):

        # get real coords from inds
        this_gamma = gamma_coord[i_gamma]
        this_beta = beta_coord[i_beta]

        # get conjugate coordinates and indices
        conj_gamma = -this_gamma
        conj_beta = (this_beta + 2*this_gamma + np.pi)%(2*np.pi)  # between (0,2pi)
        i_conj_gamma = (conj_gamma + gamma_max)/dgamma
        i_conj_beta = conj_beta/dbeta_proj

        # get difference in number of rows, depends on difference in beta
        this_dproj = (conj_beta-this_beta)/dbeta_proj
        this_drows = int(this_dproj*dz_proj/dvz)

        # get integers/weights for linear interp
        ic_gamma, ic_beta = int(i_conj_gamma), int(i_conj_beta)
        w_gamma, w_beta = i_conj_gamma%1, i_conj_beta%1

        # get conjugate ray with linear interp for beta and gamma
        if ic_beta<N_proj_rot-1 and ic_gamma<cols-1:

            conj_local = q_orig[ic_beta:ic_beta+2, :, ic_gamma:ic_gamma+2]
            beta_sum = (1-w_beta)*conj_local[0] + w_beta*conj_local[1]
            gamma_sum = (1-w_gamma)*beta_sum[:,0] + w_gamma*beta_sum[:,1]
        else:
            gamma_sum = q_orig[ic_beta, :, ic_gamma]

        if this_drows>0:
            q_conjugate[:-this_drows, i_gamma] = gamma_sum[this_drows:]
        else:
            q_conjugate[-this_drows:, i_gamma] = gamma_sum[:this_drows]

    return q_conjugate


def do_conjugate_ray_weighting(data_beta, w3D, cols, gamma_coord, vz_coord, dbeta_proj, dz_proj):
    
    result = np.zeros(data_beta.shape)
    for i_beta in range(len(data_beta)):
        
        # get conjugate projection
        proj_conj = get_conjugate_proj(i_beta, data_beta, cols, gamma_coord, vz_coord, dbeta_proj, dz_proj)
        
        # get weights for original projection and conjugate projection
        w3D_rays = np.tile(np.reshape(w3D, [rows,1]), [N_rot, cols])
        w3D_conj = 1 - w3D_rays
        
        # do the weighted sum
        result[i_beta] = w3D_rays*data_beta[i_beta] + w3D_conj*proj_conj
        
    return result
