#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:07:07 2022

@author: gjadick
"""

import numpy as np

from time import time

import matplotlib.pyplot as plt
from datetime import datetime


def get_angle(x,y):
    ''' calcs angle in x-y plane, shifts to range (0, 2pi)'''
    theta = np.arctan(y/x)
    if x < 0:  # quadrants 2,3
        theta += np.pi
    elif y < 0: # quadrant 4
        theta += 2*np.pi
    return theta


def get_L2(r, theta, beta, SID):
    '''calcs rescale value for projection'''
    return r**2 * np.cos(beta - theta)**2 + (SID + r*np.sin(beta - theta))**2


def get_gamma(r, theta, beta, SID):
    '''calcs the gamma passing through given matrix (x,y) coordinate '''
    return np.arctan(r*np.cos(beta - theta)/(SID + r*np.sin(beta - theta)))
    #return get_angle(SID + r*np.sin(beta - theta), r*np.cos(beta - theta))


def get_gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))


def get_z_data(z_target, z_width, z, img):
    ''' 
    Truncates a full image to the desired z_width.
    Considers z_width as Gaussian FWHM , then sums over all z for each column.
    inputs:
     - z_target: target z coordinate for the recon slice [mm]
     - z_width: target thickness for the recon slice [mm], centered about z_target
     - z: coordinates corresponding to the projection data [mm]
     - img: 2D matrix of fan beam projections, with first index corresponding to height z. Must have shape [len(z), cols].
    '''

    z_target_sigma = z_width / (2*np.sqrt(2*np.log(2)))  # FWHM -> sigma for Gaussian
    z_weights = get_gaussian(z, z_target, z_target_sigma)
    z_weights /= np.trapz(z_weights, x=z)  # normalize
    weighted_image = (img.T * z_weights[::-1]).T
    return np.sum(weighted_image, axis=0)   


def do_recon(q_filtered,                                       # projections
             SID, dbeta_proj, dz_proj, gamma_coord, vz_coord,  # projection params
             z_target, z_width, N_matrix, FOV,                 # recon params
             verbose=True):

    ## initialize recon matrix coordinates, faster calculating
    sz = FOV/N_matrix 
    matrix_coord_1d = np.arange((1-N_matrix)*sz/2, N_matrix*sz/2, sz)

    X_matrix, Y_matrix = np.meshgrid(matrix_coord_1d, -matrix_coord_1d)
    r_matrix = np.sqrt(X_matrix**2 + Y_matrix**2)
    theta_matrix = np.reshape([get_angle(X_matrix.ravel()[i], Y_matrix.ravel()[i]) for i in range(N_matrix**2)], [N_matrix, N_matrix])

    # target coordinates
    ji_coord = np.reshape(np.array(np.meshgrid(range(N_matrix),range(N_matrix))).T, [N_matrix**2, 2])

    ## iterate over each beta 
    matrix = np.zeros([N_matrix, N_matrix])
    t0 = time()
        
    for i_beta, proj in enumerate(q_filtered):

        if verbose:
            if i_beta%100 == 0:
                print(f'{z_target:8.3f} mm, {100*i_beta/len(q_filtered):5.1f}%: {time()-t0:.3f} s')

        # get the coordinates
        beta = i_beta*dbeta_proj            
        this_gamma = gamma_coord
        this_vz = i_beta*dz_proj + vz_coord

        # isolate the z for this slice
        proj_z = get_z_data(z_target, z_width, this_vz, proj)
        
        # calculate matrix with target gamma and L^2 factors
        gamma_target_matrix = get_gamma(r_matrix, theta_matrix, beta, SID)
        L2_matrix = get_L2(r_matrix, theta_matrix, beta, SID)
            
        for j,i in ji_coord:
            this_q = np.interp(gamma_target_matrix[j,i], this_gamma, proj_z)
            matrix[j,i] += this_q * dbeta_proj / L2_matrix[j,i]     
        
    fig,ax=plt.subplots(dpi=300)
    m = ax.imshow(matrix, cmap='gray')
    plt.colorbar(m)
    plt.title(f'z = {z_target:.3f} mm, width = {z_width:.3f} mm')
    now = datetime.now()
    plt.savefig(f'output/test_{now.strftime("%Y_%m_%d_%H_%M_%S")}.png')
    plt.show()
        
    return matrix
