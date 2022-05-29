#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:06:36 2022

@author: gjadick
"""

import os 
import numpy as np
from time import time
#from datetime import datetime
#import multiprocessing as mp

from file_manager import read_dcm_proj, make_output_dir, img_to_dcm
from preprocess import get_G, get_w3D
from fbp import get_recon_coords, get_sinogram, do_recon    
from postprocess import get_HU

import matplotlib.pyplot as plt
        
if __name__=='__main__':

    #data_dir =  'reference'
    #proj_dir =  os.path.join(data_dir, 'dcmproj_reference')
    #recon_dir = os.path.join(data_dir, 'dcmrecon_reference')
    
    proj_dir =  'input/dcmproj_liver/dcm_134'
    #proj_dir = 'input/dcmproj_lung_lesion/dcm_067/'
    ######################################################################
    ######################################################################
    
    ### ACQUISITION PARAMS
    
    N_proj_rot = 1000 
    rows = 64
    cols = 900
    sz_row = 1.0         # mm
    sz_col = 1.0         # mm
    
    kVp = 120.0          # kV
    FSy, FSz = 1.0, 1.0  # mm
    BC = 35.05           # mm, beam collimation at isocenter
    SID = 575.0          # mm, source-isocenter distance 
    SDD = 1050.0         # mm, source-detector distance 
    pitch = 1.0          # ratio dz_per_rot / BC
    
    ### RECON PARAMS
    
    FOV = 500.0          # mm
    N_matrix = 512//8       # number pixels in x,y of matrix
    z_width = 0.5467     # mm, chest (smallest possible, 35.05/64)
    #z_width = 1.5        # mm, liver
    z_targets = [100]    # mm
    
    ramp_percent = 0.85  # FT filtering of projection data, use your discretion
    kl = 1.0
    do_cone_filter = kl > 0
        
    s = sz_col*SID/SDD    # sampling distance
    fN = 1/(2*s)          # Nyquist frequency
    fc = fN*ramp_percent  # cutoff frequency, percentage of fN
    
    ### DEBUG PARAMS
    check_sinograms = True
    save_sinograms = True
    
    ######################################################################
    ######################################################################
    
    ### READ DATA
    t0 = time()
    data = read_dcm_proj(proj_dir)
    N_proj, rows, cols = data.shape
    N_rot = N_proj//N_proj_rot
    dz_rot = pitch*BC
    print(f'[{time()-t0:.1f} s] data read, {cols} cols x {rows} rows x {N_proj} proj')
    
    ### GET COORDINATES
    # beta: global angle to central row of projection
    dbeta_proj = 2*np.pi/N_proj_rot   
    
    # z: global height to row
    dz_proj = dz_rot/N_proj_rot
    z0_rot = np.arange(0, N_rot*dz_rot, dz_rot)  # initial z for each rotation

    # gamma: angle between projection vertical axis & channel center
    gamma_max = sz_col*(0.5*(cols-1))/SDD
    dgamma =    sz_col/SDD 
    gamma_coord = np.array([-gamma_max + i*dgamma for i in range(cols)])
    
    # v: height between projection horizontal axis & row center
    v_max = (SID/SDD)*sz_row*(0.5*(rows-1))
    dv    = (SID/SDD)*sz_row
    v_coord = np.array([v_max - j*dv for j in range(rows)])
 
    
    ### GROUP PROJECTIONS BY BETA
    ### (vertically stacked "mega-projections" which include all rotations)
    t0 = time()
    temp = np.reshape(data, [N_rot, N_proj_rot, rows, cols])
    data_beta = np.array([np.vstack(temp[::-1,i_beta,:,:]) for i_beta in range(N_proj_rot)])
    del data
    del temp
    
    # vz: global height of each row for mega-projection.
    vz_mesh = np.meshgrid(v_coord[::-1], z0_rot)   # [::-1] to stack in ascending order
    vz_coord = vz_mesh[0].flatten() + vz_mesh[1].flatten()
    print(f'[{time()-t0:.1f} s] projections grouped by beta')


    ### FILTER DATA
    # alpha: cone angle for each row
    t0 = time()
    if do_cone_filter:
        alpha_coord = np.array([(j - rows//2 + 0.5) * sz_row for j in range(rows)])/SDD
        w3D = get_w3D(alpha_coord, np.max(alpha_coord), kl)
        data_beta_flat = np.reshape([rotproj*np.tile(w3D,[cols,N_rot]).T for rotproj in data_beta], [N_proj*rows, cols])
    else:
        data_beta_flat = np.reshape(data_beta, [N_proj*rows, cols])
    del data_beta
    
    # G: fanbeam ramplike filter 
    G = get_G(gamma_coord, cols, s, fc)
    qm_flat = np.array([0.5*q*SID*np.cos(gamma_coord) for q in data_beta_flat])    # with cosine weighting
    qm_flat_filtered = np.array([np.fft.irfft(G*np.fft.rfft(qm)) for qm in qm_flat])
    del qm_flat
    q_filtered = np.reshape(qm_flat_filtered,  [N_proj_rot, N_rot*rows, cols])
    del qm_flat_filtered
    print(f'[{time()-t0:.1f} s] projection data preprocessed')

    
    ######################################################################
    ######################################################################
    
    t0 = time()
    
    output_dir = make_output_dir(proj_dir) 
    
    # get recon matrix coordinates
    ji_coord, r_M, theta_M, gamma_target_M, L2_M = get_recon_coords(N_matrix, FOV, N_proj_rot, dbeta_proj, SID)
    
    for i_target, z_target in enumerate(z_targets):
    
        # get sinograms
        sino = get_sinogram(q_filtered, dz_proj, vz_coord, z_target, z_width)        # data sinogram
        if do_cone_filter:
            w3D_rays = np.tile(np.reshape(w3D, [1,rows,1]), [N_proj_rot, N_rot, cols])
            w_sino = get_sinogram(w3D_rays, dz_proj, vz_coord, z_target, z_width)       # weights
            del w3D
            del w3D_rays
        else:
            w_sino = np.ones(sino.shape, dtype=np.float32)
    
        # check sinograms
        if check_sinograms:
            fig,ax=plt.subplots(1,2,dpi=150,figsize=[8,3])
            ax[0].imshow(sino)
            ax[1].imshow(w_sino)
            plt.show()
            
        if save_sinograms:
            sino.tofile('output/test_sinogram.npy')
            w_sino.tofile('output/test_weights.npy')

        # recon
        recon = do_recon(sino, w_sino, dbeta_proj, gamma_coord,                  
                     r_M, theta_M, gamma_target_M, L2_M, ji_coord,
                     verbose=True)
        
        # convert units
        recon_HU = get_HU(recon)  
        
        # save image
        filename = os.path.join(output_dir, f'{i_target+1:03}.dcm')
        img_to_dcm(recon_HU, filename, z_width, z_target, ramp_percent, kl)
                   

    #args = [(q_filtered, SID, dbeta_proj, dz_proj, gamma_coord, vz_coord, z_target, z_width, N_matrix, FOV) for z_target in z_targets]
    #with mp.Pool(5) as pool:    # multiprocessing
    #    pool.starmap(do_recon, args)
    
    # for z_target in z_targets:
    #     matrix_z = do_recon(q_filtered, SID, dbeta_proj, dz_proj, gamma_coord, vz_coord, 
    #                         z_target, z_width, N_matrix, FOV, verbose=True)
    
    print(f'[{time()-t0:.1f} s] images reconstructed')

    
    # save output
    
