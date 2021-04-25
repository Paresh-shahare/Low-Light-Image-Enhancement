import os
from PIL import Image 
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

path_low = r'/home/wireshark/Documents/CDAC/Project/LOLdataset_resized/train/low'
path_high = r'/home/wireshark/Documents/CDAC/Project/LOLdataset_resized/train/high'
path_out = r'/home/wireshark/Documents/CDAC/Project/LOLdataset_resized/train/concat'


low_file = os.listdir(path_low)
high_file = os.listdir(path_high)


for i in range(len(low_file)):
    try:
        os.chdir(path_low)
        low_img = Image.open(os.path.join(path_low, low_file[i]))
        os.chdir(path_high)
        high_img = Image.open(os.path.join(path_high, high_file[i]))
        gan_image = np.concatenate((low_img, high_img), axis=1)
        cv2.imwrite(os.path.join(path_out,high_file[i]),gan_image)
        # gan_image.save(os.path.join(path_out,high_file[i]))
        print('try')
        print(low_file[i])
    except IOError:
        print('except')
        print(low_file[i])
        pass