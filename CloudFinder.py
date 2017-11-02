#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:08:12 2017

@author: Donovan Colton

@program: CloudFinder.py
"""

from scipy import misc
import os, sys
import matplotlib.pyplot as plt
import time

files = [f for f in os.listdir("./Camera/19106/") if os.path.exists]
camerapath = os.path.join(os.getcwd(), 'Camera/19106/')
maskpath = os.path.join(os.getcwd(), 'Camera/Mask/')

#860 images in ./Cameras/1960 picking up a .DS_Store file
#at index 0 for some reason:: loop from 0 : 861

print(maskpath)
sys.exit("stop")

for file in range(10):
    try:
        
        filename = files[file]
        if (filename == ".DS_Store"):
            continue
        
        print (filename)
        
        img = misc.imread(camerapath + files[file])
        
        plt.figure(file)
        plt.imshow(img)
        time.sleep(1)
        
    except Exception as e:
        print(e)