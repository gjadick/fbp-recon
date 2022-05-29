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


def get_sinogram(q_filtered, dz_proj, vz_coord, z_target, z_width):
    '''Get the sinogram specific to the target z and width'''
    Nz, Ny, Nx = q_filtered.shape
    result = np.empty([Nz, Nx], dtype=np.float32) #152 ms ± 408 µs 
        
    for i_beta, proj in enumerate(q_filtered):
        this_vz = i_beta*dz_proj + vz_coord
        proj_z = get_z_data(z_target, z_width, this_vz, proj)
        result[i_beta] = proj_z
    return np.array(result)


def get_recon_coords(N_matrix, FOV, N_proj_rot, dbeta_proj, SID):
    '''Get the coordinates needed for the reconstruction matrix (common to all recons with same matrix)'''
    ## matrix coordinates: (r, theta)
    sz = FOV/N_matrix 
    matrix_coord_1d = np.arange((1-N_matrix)*sz/2, N_matrix*sz/2, sz)
    X_matrix, Y_matrix = np.meshgrid(matrix_coord_1d, -matrix_coord_1d)
    r_matrix = np.sqrt(X_matrix**2 + Y_matrix**2)
    theta_matrix = np.reshape([get_angle(X_matrix.ravel()[i], Y_matrix.ravel()[i]) for i in range(N_matrix**2)], [N_matrix, N_matrix])

    # fan-beam rescaling for each beta: L^2(r,theta), gamma(r,theta)
    gamma_target_matrix_all = np.empty([N_proj_rot, N_matrix, N_matrix], dtype=np.float32)
    L2_matrix_all = np.empty([N_proj_rot, N_matrix, N_matrix], dtype=np.float32)
    for i_beta in range(N_proj_rot):
        beta = i_beta*dbeta_proj            
        gamma_target_matrix_all[i_beta] = get_gamma(r_matrix, theta_matrix, beta, SID)
        L2_matrix_all[i_beta] = get_L2(r_matrix, theta_matrix, beta, SID)

    # recon matrix indices
    ji_coord = np.reshape(np.array(np.meshgrid(range(N_matrix),range(N_matrix))).T, [N_matrix**2, 2])

    return ji_coord, r_matrix, theta_matrix, gamma_target_matrix_all, L2_matrix_all
    
def lerp(v0, v1, t):
    '''linear interp'''
    return (1-t)*v0 + t*v1


def do_recon(sinogram, w_sinogram, dbeta_proj, gamma_coord,                  
             r_matrix, theta_matrix, gamma_target_matrix_all, L2_matrix_all, ji_coord,
             verbose=False):
    '''
    Main reconstruction program. Reconstructs the sinogram and normalizes by the provided weights.

    Parameters
    ----------
    sinogram : 2D matrix
        Pre-processed sinogram for reconstruction.
    w_sinogram : 2D matrix
        Matrix of weights which must be normalized out.
    dbeta_proj : float
        Change in beta angle for each projection [rad].
    gamma_coord : 1D array
        Local angle coordinate for each column [rad].
    r_matrix : 2D array
        polar r coordinate for each (i,j) to recon.
    theta_matrix : 2D array
        polar theta coordinate for each (i,j) to recon.
    gamma_target_matrix_all : 3D array
        gamma targets for linear interpolation for each (i,j,beta)
    L2_matrix_all : 3D array
        L^2 normalization factors for each (i,j,beta)
    ji_coord : 2D array
        List of the [j,i] coordinates (corresponding to y, x in recon matrix)
    verbose : bool, optional
        whether to print timing. The default is False.

    Returns
    -------
    2D matrix
        the reconstruction.

    '''
    t0 = time()

    matrix = np.zeros(r_matrix.shape)    
    w_matrix = np.zeros(r_matrix.shape)  
    
    for i_beta in range(len(sinogram)):
        proj_z = sinogram[i_beta] # fan-beam data at this z
        w_z = w_sinogram[i_beta]  # weights at this z
        
        if verbose:
            if i_beta%100 == 0:
                print(f' {100*i_beta/len(sinogram):5.1f}%: {time()-t0:.3f} s')
        
        L2_matrix = L2_matrix_all[i_beta]         # matrix with L^2 factors
        gamma_max  = np.max(gamma_coord)
        dgamma = gamma_coord[1]-gamma_coord[0]
        for j,i  in ji_coord:
            gamma_target = gamma_target_matrix_all[i_beta,j,i]
            if np.abs(gamma_target) >= gamma_max:
                pass
            else:
                i_gamma0 = int((gamma_target + gamma_max)//dgamma)
                t = (dgamma*(i_gamma0+1) - gamma_max - gamma_target)/dgamma
                this_q = lerp(proj_z[i_gamma0], proj_z[i_gamma0 + 1], t)
                this_w = lerp(w_z[i_gamma0], w_z[i_gamma0 + 1], t)
                matrix[j,i] += this_q * dbeta_proj  / L2_matrix[j,i]  
                w_matrix[j,i] += this_w  / L2_matrix[j,i]  

    return matrix/w_matrix

        
    fig,ax=plt.subplots(dpi=300)
    m = ax.imshow(matrix, cmap='gray')
    plt.colorbar(m)
    now = datetime.now()
    plt.savefig(f'output/test_{now.strftime("%Y_%m_%d_%H_%M_%S")}.png')
    plt.show()
        
    return matrix
