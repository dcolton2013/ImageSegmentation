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
import numpy as np

#Sky pixels arrays
#150
cloudy_test_X = np.empty((0,3), int)
cloudy_test_y = np.empty((0,1), int)

#131
clear_test_X  = np.empty((0,3), int)
clear_test_y = np.empty((0,1), int)

avgBlueClear = []
avgBlueCloudy = []

#def avgBlueness(array):  
#    avgs = []
#    for i in range(np.shape(array)[0]):
#        avg = 0        
#        for j in range( np.shape( array[i] )[0] ):
#            avg = 0
#            redPixel = array[i][j][0]
#            greenPixel = array[i][j][1]
#            bluePixel = array[i][j][2]
#            avg += bluePixel - redPixel
#            avg += bluePixel - greenPixel
#        #avgs.append(float( avg/ np.shape( array[i] )[0] ))
#    return avgs       
        
def cloudyImages(cloudy,path,mask):
    temp_X = np.empty((0,3), int)
    
    for file in cloudy:
        try:
            if (file == ".DS_Store"):
                continue
            
            #print (file)
            img = misc.imread(path+file)
            resized_img = misc.imresize(img,(70,80)) 
            resized_img.flatten()    
            
            temp_X = np.concatenate( (temp_X,resized_img[mask==255]) )
            
            #plt.figure(file)
            #plt.imshow(resized_img)

            
        except Exception as e:
            #error reading image
            print(e)
    

    return temp_X
               
def clearImages(clear,path,mask):
    temp  = np.empty((0,3), int)
    for file in clear:
        try:
            if (file == ".DS_Store"):
                continue
            #print (file)
            img = misc.imread(path+file)
            resized_img = misc.imresize(img,(70,80)) 
            resized_img.flatten();
                               
            temp = np.concatenate( (temp,resized_img[mask==255] ) )
            
            #plt.figure(file)
            #plt.imshow(img)
            
        except Exception as e:
            #error reading image
            print(e)
    return temp

############## 4232
cloudy4232 = [f for f in os.listdir("./Cameras/4232/cloudy") if os.path.exists]
clear4232 = [f for f in os.listdir("./Cameras/4232/clear") if os.path.exists]
mask4232 = misc.imread('./Cameras/4232/mask/4232.png')
mask4232 = misc.imresize(mask4232,(70,80),interp = 'nearest')
path_cloudy_4232 = os.path.join(os.getcwd(), 'Cameras/4232/cloudy/')
path_clear_4232 = os.path.join(os.getcwd(), 'Cameras/4232/clear/')

cloudy_test_X= np.concatenate( (cloudy_test_X,cloudyImages(cloudy4232,path_cloudy_4232,mask4232) ))
clear_test_X = np.concatenate( (clear_test_X,clearImages(clear4232,path_clear_4232,mask4232) ))
##############        

############## 7371
cloudy7371 = [f for f in os.listdir("./Cameras/7371/cloudy") if os.path.exists]
clear7371 = [f for f in os.listdir("./Cameras/7371/clear") if os.path.exists]
mask7371 = misc.imread('./Cameras/7371/mask/7371.png')
mask7371 = misc.imresize(mask7371,(70,80),interp = 'nearest')
path_cloudy_7371 = os.path.join(os.getcwd(), 'Cameras/7371/cloudy/')
path_clear_7371 = os.path.join(os.getcwd(), 'Cameras/7371/clear/')

cloudy_test_X = np.concatenate( (cloudy_test_X,cloudyImages(cloudy7371,path_cloudy_7371,mask7371) ))
clear_test_X = np.concatenate( (clear_test_X,clearImages(clear7371,path_clear_7371,mask7371) ))
##############

############## 9112
cloudy9112 = [f for f in os.listdir("./Cameras/9112/cloudy") if os.path.exists]
clear9112 = [f for f in os.listdir("./Cameras/9112/clear") if os.path.exists]
mask9112 = misc.imread('./Cameras/9112/mask/9112.png')
mask9112 = misc.imresize(mask9112,(70,80),interp = 'nearest')
path_cloudy_9112 = os.path.join(os.getcwd(), 'Cameras/9112/cloudy/')
path_clear_9112 = os.path.join(os.getcwd(), 'Cameras/9112/clear/')

cloudy_test_X = np.concatenate( (cloudy_test_X,cloudyImages(cloudy9112,path_cloudy_9112,mask9112) ))
clear_test_X = np.concatenate( (clear_test_X,clearImages(clear9112,path_clear_9112,mask9112) ))
##############

############## 10870
cloudy10870 = [f for f in os.listdir("./Cameras/10870/cloudy") if os.path.exists]
clear10870 = [f for f in os.listdir("./Cameras/10870/clear") if os.path.exists]
mask10870 = misc.imread('./Cameras/10870/mask/10870.png')
mask10870 = misc.imresize(mask10870,(70,80),interp = 'nearest')
path_cloudy_10870 = os.path.join(os.getcwd(), 'Cameras/10870/cloudy/')
path_clear_10870 = os.path.join(os.getcwd(), 'Cameras/10870/clear/')

cloudy_test_X = np.concatenate( (cloudy_test_X,cloudyImages(cloudy10870,path_cloudy_10870,mask10870) ))
clear_test_X = np.concatenate( (clear_test_X,clearImages(clear10870,path_clear_10870,mask10870) ))
##############

############## 19106
cloudy19106 = [f for f in os.listdir("./Cameras/19106/cloudy") if os.path.exists]
clear19106 = [f for f in os.listdir("./Cameras/19106/clear") if os.path.exists]
mask19106 = misc.imread('./Cameras/19106/mask/19106.png')
mask19106 = misc.imresize(mask19106,(70,80),interp = 'nearest')
path_cloudy_19106 = os.path.join(os.getcwd(), 'Cameras/19106/cloudy/')
path_clear_19106 = os.path.join(os.getcwd(), 'Cameras/19106/clear/')

cloudy_test_X = np.concatenate( (cloudy_test_X,cloudyImages(cloudy19106,path_cloudy_19106,mask19106) )) 
clear_test_X = np.concatenate( (clear_test_X,clearImages(clear19106,path_clear_19106,mask19106) ))
##############

cloudy_test_y = np.ones(np.shape(cloudy_test_X)[0])
clear_test_y = np.zeros(np.shape(clear_test_X)[0])

#calculate avg blueness array
#avgBlueCloudy = avgBlueness(cloudy_test_X)
#avgBlueClear = avgBlueness(clear_test_X)

