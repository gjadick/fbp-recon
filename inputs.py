#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:06:58 2022

@author: gjadick
"""

import os
import pydicom

def read_dcm_proj(proj_dir):
    '''
    Reads the Dicom projection files into a single 3D numpy array.

    Parameters
    ----------
    proj_dir : str
        path to directory with the .dcm projection files.

    Returns
    -------
    data : numpy array, float32 [N_proj, rows, cols]
        3D array of projection data.

    '''
    # get sorted list of .dcm files
    proj_files = sorted([x for x in os.listdir(proj_dir) if x[-4:]=='.dcm'])
    N_proj = len(proj_files)
    
    # get rows and columns from first file
    ds = pydicom.dcmread(os.path.join(proj_dir, proj_files[0]))
    rows = int(ds.Rows)
    cols = int(ds.Columns)
                        
    # read in all projections
    data = np.empty([N_proj, rows, cols], dtype=np.float32) 
    for ind, file_name in enumerate(proj_files):

        this_proj_file = os.path.join(proj_dir, file_name)
        ds = pydicom.dcmread(this_proj_file)
    
        # get matrix and rescale intensities to ln(I0/I) = mu*L values
        m = float(ds.RescaleSlope)
        b = float(ds.RescaleIntercept)
        this_proj = m*ds.pixel_array.T + b   # transpose so columns ~ x-axis
        try:
            data[ind] = this_proj
        except:
            print(f"projection shape mismatch for {ind}, {this_proj_file}")
    
    return data