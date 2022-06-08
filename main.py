#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:06:36 2022

@author: gjadick
"""

import sys
import os 
import numpy as np
from time import time
from datetime import datetime

from file_manager import read_dcm_proj, make_output_dir, img_to_dcm
from preprocess import get_G, get_w3D, do_conjugate_ray_weighting
from fbp import get_recon_coords, get_sinogram, do_recon, do_recon_weights    
from fbp_gpu import do_recon_gpu, do_recon_weights_gpu
from postprocess import get_HU

import matplotlib.pyplot as plt


##########################################################################

def main(proj_dir, z_width, ramp_percent, kl, detail_mode=False, verbose=False, conjugate_weighting=False):
 
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
    
    if conjugate_weighting:
        data_beta_flat = np.reshape(do_conjugate_ray_weighting(data_beta, w3D, rows, cols, N_rot, gamma_coord, vz_coord, dbeta_proj, dz_proj), [N_proj*rows, cols])

    else:
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
        
        # get sinogram
        sino = get_sinogram(q_filtered, dz_proj, vz_coord, z_target, z_width)        

        # recon
        if conjugate_weighting:
            if use_GPU:
                recon = do_recon_gpu(sino, gamma_target_M, L2_M, gamma_coord, dbeta_proj)
            else:
                recon = do_recon(sino, dbeta_proj, gamma_coord,      
                     gamma_target_M, L2_M, ji_coord, verbose=verbose)
            recon_HU = get_HU(recon, 0.0525587, -0.0047017)  

        else:
            w3D_rays = np.tile(np.reshape(w3D, [1,rows,1]), [N_proj_rot, N_rot, cols])
            w_sino = get_sinogram(w3D_rays, dz_proj, vz_coord, z_target, z_width)       # weights
            if use_GPU:
                recon = do_recon_weights_gpu(sino, w_sino, 
                    gamma_target_M, L2_M, gamma_coord, dbeta_proj)
            else:
                recon = do_recon_weights(sino, w_sino, dbeta_proj, gamma_coord,      
                     gamma_target_M, L2_M, ji_coord,
                     verbose=verbose)
            recon_HU = get_HU(recon, 9.528875, -1.0764065)

        # save image
        filename = os.path.join(output_dir, f'{i_target+1:04}.dcm')
        img_to_dcm(recon_HU, filename, z_width, z_target, ramp_percent, kl)
        if verbose:
            print(f'\t{filename} finished')

    if verbose:               
        print(f'[{time()-t0:.1f} s] images reconstructed')

    
##########################################################################

if __name__=='__main__':

    organ = sys.argv[3] # must be liver, lung, copd

    if organ=='liver':
        main_dir = 'input/dcmproj_liver'
        z_width = 1.5
        ramp_percent = 0.50
        kl = 0.50
        detail_mode = False
        conjugate_mode =  True

    elif organ=='lung':
        main_dir = 'input/dcmproj_lung_lesion'
        z_width = 0.5467
        ramp_percent = 0.60
        kl = 1.0 
        detail_mode = True
        conjugate_mode =  True

    elif organ=='copd':
        main_dir = 'input/dcmproj_copd'
        z_width = 0.5467
        ramp_percent = 0.90
        kl = 1.0 
        detail_mode = True
        conjugate_mode =  True
    
    else: 
        print(f'organ {organ} not valid argument')


    case_files = sorted([x for x in os.listdir(main_dir) if 'dcm_' in x]) 
    N_tot = len(case_files)

    # for multiple GPUs, divide cases into equal fractions
    this_frac = int(sys.argv[1])
    run_fracs = int(sys.argv[2])
    case_files = case_files[int(N_tot*(this_frac-1)/run_fracs):int(N_tot*this_frac/run_fracs)] 
    
    for case_id in case_files:
        proj_dir = os.path.join(main_dir, case_id)
        now = datetime.now()
        print(f'\n[{now.strftime("%Y_%m_%d_%H_%M_%S")}] {proj_dir}')
        main(proj_dir, z_width, ramp_percent, kl, detail_mode, conjugate_weighting=conjugate_mode)

