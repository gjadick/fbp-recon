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
from fbp_gpu import do_recon_gpu
from postprocess import get_HU

import matplotlib.pyplot as plt


##########################################################################

def main(proj_dir, z_width, ramp_percent, kl, detail_mode=False, verbose=False):
 
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
    
    use_GPU = True

    FOV = 500.0          # mm
    N_matrix = 512       # [N,N] pixels in recon matrix
    
    s = sz_col*SID/SDD    # sampling distance
    fN = 1/(2*s)          # Nyquist frequency
    fc = fN*ramp_percent  # cutoff frequency, percentage of fN
    
    ### DEBUG PARAMS
    check_sinograms = False
    save_sinograms = False
    
    ######################################################################
    
    ### READ DATA
    t0 = time()
    data = read_dcm_proj(proj_dir)
    N_proj, rows, cols = data.shape
    N_rot = N_proj//N_proj_rot
    dz_rot = pitch*BC
    if verbose:
        print(f'[{time()-t0:.1f} s] data read, {cols} cols x {rows} rows x {N_proj} proj')
   
    # assign z_targets for a full scan
    z_targets = np.arange(BC, (N_rot-1)*BC, z_width) + z_width/2
    print(f'Target z assigned: {len(z_targets)} slices to recon, {z_targets[0]:.3f} mm to {z_targets[-1]:.3f} mm')
    
    ### GET COORDINATES
    # beta: global angle to central row of projection
    dbeta_proj = 2*np.pi/N_proj_rot   
    
    # z: global height to row
    dz_proj = dz_rot/N_proj_rot
    z0_rot = np.arange(0, N_rot*dz_rot, dz_rot,dtype=np.float32)  # initial z for each rotation

    # gamma: angle between projection vertical axis & channel center
    gamma_max = sz_col*(0.5*(cols-1))/SDD
    dgamma =    sz_col/SDD 
    gamma_coord = np.array([-gamma_max + i*dgamma for i in range(cols)], dtype=np.float32)
    
    # v: height between projection horizontal axis & row center
    v_max = (SID/SDD)*sz_row*(0.5*(rows-1))
    dv    = (SID/SDD)*sz_row
    v_coord = np.array([v_max - j*dv for j in range(rows)], dtype=np.float32)
 
    
    ### GROUP PROJECTIONS BY BETA
    ### (vertically stacked "mega-projections" which include all rotations)
    t0 = time()
    temp = np.reshape(data, [N_rot, N_proj_rot, rows, cols])
    data_beta = np.array([np.vstack(temp[::-1,i_beta,:,:]) for i_beta in range(N_proj_rot)], dtype=np.float32)
    del data
    del temp
    
    # vz: global height of each row for mega-projection.
    vz_mesh = np.meshgrid(v_coord[::-1], z0_rot)   # [::-1] to stack in ascending order
    vz_coord = vz_mesh[0].flatten() + vz_mesh[1].flatten()
    vz_coord = np.float32(vz_coord)
    if verbose:
        print(f'[{time()-t0:.1f} s] projections grouped by beta')


    ### FILTER DATA
    # alpha: cone angle for each row
    t0 = time()
    alpha_coord = np.array([(j - rows//2 + 0.5) * sz_row for j in range(rows)], dtype=np.float32)/SDD
    w3D = get_w3D(alpha_coord, np.max(alpha_coord), kl, detail_mode=detail_mode)
    data_beta_flat = np.reshape([rotproj*np.tile(w3D,[cols,N_rot]).T for rotproj in data_beta], [N_proj*rows, cols])
    del data_beta
    
    # G: fanbeam ramplike filter 
    G = get_G(gamma_coord, cols, s, fc)
    qm_flat = np.array([0.5*q*SID*np.cos(gamma_coord) for q in data_beta_flat], dtype=np.float32)    # with cosine weighting
    qm_flat_filtered = np.array([np.fft.irfft(G*np.fft.rfft(qm)) for qm in qm_flat], dtype=np.float32)
    del qm_flat
    q_filtered = np.reshape(qm_flat_filtered,  [N_proj_rot, N_rot*rows, cols])
    del qm_flat_filtered
    if verbose:
        print(f'[{time()-t0:.1f} s] projection data preprocessed')

    
    ######################################################################
    
    t0 = time()
    
    output_dir = make_output_dir(proj_dir) 
    
    # get recon matrix coordinates
    ji_coord, r_M, theta_M, gamma_target_M, L2_M = get_recon_coords(N_matrix, FOV, N_proj_rot, dbeta_proj, SID)
    
    for i_target, z_target in enumerate(z_targets):
        print(f'[{i_target+1:03}/{len(z_targets):03}] {z_target:.3f} mm, {time()-t0:.1f} s') 
        
        filename = os.path.join(output_dir, f'{i_target+1:03}.dcm')
        # get sinograms
        sino = get_sinogram(q_filtered, dz_proj, vz_coord, z_target, z_width)        # data sinogram
        w3D_rays = np.tile(np.reshape(w3D, [1,rows,1]), [N_proj_rot, N_rot, cols])
        w_sino = get_sinogram(w3D_rays, dz_proj, vz_coord, z_target, z_width)       # weights
    
        # check sinograms
        if check_sinograms:
            fig,ax=plt.subplots(1,2,dpi=150,figsize=[8,3])
            ax[0].imshow(sino)
            ax[1].imshow(w_sino)
            plt.show()
            
        if save_sinograms:
            np.save('output/test_sinogram.npy', sino)
            np.save('output/test_weights.npy', w_sino)

        # recon
        if use_GPU:
            recon = do_recon_gpu(sino, w_sino, 
                    gamma_target_M, L2_M, gamma_coord, dbeta_proj)
        else:
            recon = do_recon(sino, w_sino, dbeta_proj, gamma_coord,      
                     gamma_target_M, L2_M, ji_coord,
                     verbose=verbose)
        
        # convert units
        recon_HU = get_HU(recon)  
        
        # save image
        filename = os.path.join(output_dir, f'{i_target+1:03}.dcm')
        img_to_dcm(recon_HU, filename, z_width, z_target, ramp_percent, kl)
        if verbose:
            print(f'\t{filename} finished')

    if verbose:               
        print(f'[{time()-t0:.1f} s] images reconstructed')

    
##########################################################################

if __name__=='__main__':

    organ = 'liver' # must be liver, lung, copd

    if organ=='liver':
        main_dir = 'input/dcmproj_liver'
        z_width = 1.5
        ramp_percent = 0.40
        kl = 0.40
        detail_mode = False

    elif organ=='lung':
        main_dir = 'input/dcmproj_lung_lesion'
        z_width = 0.5467
        ramp_percents = 0.60
        kl = 1.0 
        detail_mode = True

    elif organ=='copd':
        main_dir = 'input/dcmproj_copd'
        z_width = 0.5467
        ramp_percents = 0.90
        kl = 1.0 
        detail_mode = True
    
    else: 
        print(f'organ {organ} not valid argument')

    case_files = sorted([x for x in os.listdir(main_dir) if 'dcm_' in x]) 
    
    for case_id in case_files:
        proj_dir = os.path.join(main_dir, case_id)
        print(f'\n{proj_dir}')
        main(proj_dir, z_width, ramp_percent, kl, detail_mode)
        

