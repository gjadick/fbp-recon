#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:52:54 2022

@author: giajadick
"""

def get_HU(x, u_water=9.2793, u_air=-1.14929):

    '''
    convert to Hounsfield units. 
    u_water,u_air directly measured from  a sample recon (not physical).
    '''
    HU = 1000*(x-u_water)/(u_water-u_air)
    return HU