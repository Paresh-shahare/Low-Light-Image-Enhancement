import os
from PIL import Image 
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

path_low = r'/home/wireshark/Documents/CDAC/Project/LOLdataset/Test/Test/VV'
path_high = r'/home/wireshark/Documents/CDAC/Project/LOLdataset/Test/Test/VVcopy'
path_out = r'/home/wireshark/Documents/CDAC/Project/LOLdataset/Test/Test/out_test'


low_file = os.listdir(path_low)
high_file = os.listdir(path_high)

height = 256
width = 256

for i in range(len(low_file)):
    try:
        os.chdir(path_low)
        low_img = cv2.imread(os.path.join(path_low, low_file[i]))
        low_img = cv2.resize(low_img, (height, width),interpolation = cv2.INTER_NEAREST)
        os.chdir(path_high)
        high_img = cv2.imread(os.path.join(path_high, high_file[i]))
        high_img = cv2.resize(high_img, (height, width),interpolation = cv2.INTER_NEAREST)
        gan_image = np.concatenate((low_img, high_img), axis=1)
        cv2.imwrite(os.path.join(path_out,high_file[i]),gan_image)
        print('try')
        print(low_file[i])
    except IOError:
        print('except')
        print(low_file[i])
        pass