#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:06:36 2022

@author: gjadick
"""

import os 
import numpy as np

from inputs import read_dcm_proj
from preprocess import get_H, ramp_filter, get_w3D
from fbp import do_recon    

import matplotlib.pyplot as plt
        
if __name__=='__main__':

    #data_dir =  'reference'
    #proj_dir =  os.path.join(data_dir, 'dcmproj_reference')
    #recon_dir = os.path.join(data_dir, 'dcmrecon_reference')
    
    proj_dir =  'dcmproj_lung_lesion/dcm_067/'
    proj_files = sorted([x for x in os.listdir(proj_dir) if x[-4:]=='.dcm'])
    N_proj = len(proj_files)
    
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
    
    N_rot = N_proj//N_proj_rot
    dz_rot = pitch*BC
    
    ### RECON PARAMS
    
    FOV = 500.0          # mm
    N_matrix = 50# 512       # number pixels in x,y of matrix
    z_width = 0.5467     # mm, chest (smallest possible, 35.05/64)
    z_width = 1.5        # mm, liver
    z_targets = []
    
    ramp_percent = 0.85  # FT filtering of projection data, use your discretion
    do_cone_filter = True
    kl = 0.5 

    
    ######################################################################
    ######################################################################
    
    ### READ DATA
    data = read_dcm_proj(proj_dir)
    N_proj, rows, cols = data.shape
    
    
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
    temp = np.reshape(data, [N_rot, N_proj_rot, rows, cols])
    data_beta = np.array([np.vstack(temp[::-1,i_beta,:,:]) for i_beta in range(N_proj_rot)])

    # vz: global height of each row for mega-projection.
    vz_mesh = np.meshgrid(v_coord[::-1], z0_rot)   # [::-1] to stack in ascending order
    vz_coord = vz_mesh[0].flatten() + vz_mesh[1].flatten()


    ### FILTER DATA
    # alpha: cone angle for each row
    if do_cone_filter:
        alpha_coord = np.array([(j - rows//2 + 0.5) * sz_row for j in range(rows)])/SDD
        w3D = get_w3D(alpha_coord, np.max(alpha_coord), kl)
        data_beta_flat = np.reshape([rotproj*np.tile(w3D,[cols,N_rot]).T for rotproj in data_beta], [N_proj*rows, cols])
    else:
        data_beta_flat = np.reshape(data_beta, [N_proj*rows, cols])
        
    # ramp  
    H = get_H(ramp_percent) 
    C = 0.5*SDD*np.cos(gamma_coord)*(gamma_coord/np.sin(gamma_coord))**2
    q_flat = np.array([C*q for q in data_beta_flat])
    q_flat_filtered = np.array([ramp_filter(q, H) for q in q_flat])
    q_filtered = np.reshape(q_flat_filtered,  [N_proj_rot, N_rot*rows, cols])
    
    
    ######################################################################
    ######################################################################
    

    for z_target in z_targets:
        matrix_z = do_recon(q_filtered, dbeta_proj, dz_proj, gamma_coord, vz_coord, 
                            z_target, z_width, N_matrix, FOV, verbose=True)
    
        fig,ax=plt.subplots(dpi=300)
        m = ax.imshow(matrix_z, cmap='gray')
        plt.colorbar(m)
        plt.show()
    
    
    
    