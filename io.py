#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:06:58 2022

@author: gjadick
"""

import os
import pydicom
import numpy as np
from datetime import datetime

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
    data = np.empty([N_proj, cols, rows], dtype=np.float32) 
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


def img_to_dcm(img, filename, z_width, z_target, ramp_percent, kl, FOV=500):
    '''
    Writes the image (assumed to be CT data, float32) to filename.
    If '.dcm' extension missing from filename, it is added.

    Parameters
    ----------
    img : 2D numpy array (float32)
        reconstructed CT slice.
    filename : str
        output filename.
    z_width: float32
        slice width [mm]
    z_target: float32
        slice location (center) [mm]
    ramp_percent: float32
        percentage of Nyquist frequency used as max value in fanbeam ramplike filter
    kl: float32 
        parameter for the alpha weighting cone beam correction
        
    Returns
    -------
    None.

    '''   
    if filename[-4:]!='.dcm':
        filename+='.dcm'
    
    # initialize meta info 
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2')
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.UID("1.2.3")
    file_meta.ImplementationClassUID = pydicom.uid.UID("1.2.3.4")

    # initialize dataset using the file meta
    ds = pydicom.dataset.FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # add name, date, time
    ds.PatientName = filename.replace('.dcm','').replace('/','_')
    ds.PatientID = ""
    dt = datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S.%f')  

    # add image
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    m, b = 1.0, -4096.0
    img_uint16 = np.uint16(np.int16(img)-b)
    ds.PixelData = img_uint16

    ds.Rows = img.shape[0]
    ds.Columns = img.shape[1]
    ds.RescaleSlope = m
    ds.RescaleIntercept = b
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelRepresentation = 0
    
    ds.ReconstructionFieldOfView = FOV
    ds.PixelSpacing = [FOV/img.shape[0], FOV/img.shape[1]]
    ds.SliceThickness = z_width
    ds.SpacingBetweenSlices = z_width
    ds.SliceLocation = z_target
    
    # filter info
    ds.ImageFilter = 'weighted ramp'
    ds.ImageFilterDescription = f'Fanbeam ramp-like filter (ramp_percent={ramp_percent:.2f}), cone beam alpha weighting (kl={kl:.2f})'

    # save
    ds.save_as(filename)




