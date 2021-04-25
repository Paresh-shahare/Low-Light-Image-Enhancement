import os
import numpy as np
import cv2
import random
from PIL import Image 
import matplotlib.pyplot as plt
    
height = 256
width = 256
dataset_dir = r'/home/wireshark/Documents/CDAC/Project/Dataset/indoorCVPR_09/Images'
output_dir = r'/home/wireshark/Documents/CDAC/Project/Dataset/indoorCVPR_09/lowLight_Data'
    
original = os.listdir(dataset_dir)
os.chdir(dataset_dir)
for i in range(len(original)):
    try:
        random_scale = round(random.uniform(0.15,0.36),2)
        original_image = cv2.imread(original[i])
        Resize_original_image = cv2.resize(original_image, (256, 256),interpolation = cv2.INTER_NEAREST)
        Roriginal_image = cv2.cvtColor(Resize_original_image,cv2.COLOR_BGR2RGB)        
        hsvImg = cv2.cvtColor(Roriginal_image,cv2.COLOR_BGR2HSV)
        hsvImg[...,2] = hsvImg[...,2]*random_scale
        low_image = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)
        gan_image = np.concatenate((low_image, Resize_original_image), axis=1)
        cv2.imwrite(os.path.join(output_dir,original[i]),gan_image)
        print('try')
        print(original[i])
    except IOError:
        print('except')
        print(original[i])
        pass
    break