import os
import numpy as np
import cv2
import random
from PIL import Image 
import matplotlib.pyplot as plt
    
height = 256
width = 256
dataset_dir = r'/home/wireshark/Documents/CDAC/Project/Dataset/indoorCVPR_09/Images'
output_dir = r'/home/wireshark/Documents/CDAC/Project/Dataset/indoorCVPR_09/test_images'

directories = os.listdir(dataset_dir)
for i in range(len(directories)):
    inside_dir_path = os.path.join(dataset_dir, directories[i])
    original = os.listdir(inside_dir_path)
    os.chdir(inside_dir_path)
    # print(inside_dir_path)
    for i in range(1):
        try:
            random_scale = round(random.uniform(0.07,0.28),2)
            print(original[i])
            original_image = cv2.imread(original[i])
            Resize_original_image = cv2.resize(original_image, (256, 256),interpolation = cv2.INTER_NEAREST)
            Roriginal_image = cv2.cvtColor(Resize_original_image,cv2.COLOR_BGR2RGB)        
            hsvImg = cv2.cvtColor(Roriginal_image,cv2.COLOR_BGR2HSV)
            hsvImg[...,2] = hsvImg[...,2]*random_scale
            low_image = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)
            gan_image = np.concatenate((low_image, Resize_original_image), axis=1)
            cv2.imwrite(os.path.join(output_dir,original[i]),gan_image)
            print('try')
        except IOError:
            print('except')
            print(original[i])
            pass

output_dir = r'/home/wireshark/Documents/CDAC/Project/Dataset/indoorCVPR_09/lowLight_Data'
png_dir = r'/home/wireshark/Documents/CDAC/Project/Dataset/indoorCVPR_09/lowLightData_png'
con = os.listdir(output_dir)
os.chdir(output_dir)
from PIL import Image 
import os 
for image in range(len(con)):
    try:
        os.chdir(output_dir)
        if con[image].endswith(".jpg"):
            prefix = con[image].split(".jpg")[0]
            print(prefix)
            im = Image.open(con[image])
            os.chdir(png_dir)
            im.save(prefix + '.png')
            print('try')
            print(con[image])
        else:
            print("Inside else")
    except IOError:
        print('except')
        print(con[image])
        pass    