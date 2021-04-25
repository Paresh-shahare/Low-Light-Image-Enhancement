import os
from PIL import Image 
import random

path = r'/home/wireshark/Documents/CDAC/Project/LOLdataset_resized/our485/high'
os.chdir(path)



###  # # # random renamer

#for index, file in enumerate(files):
#    r = random.randint(101,10000)
#    os.rename(os.path.join(path, file), os.path.join(path, str(r)+'.jpg'))
    
    
    
height=input('Enter image height')
width=input('Enter image width')
files = os.listdir(path)
print(path)
# for index, file in enumerate(files):
#     os.rename(os.path.join(path, file), os.path.join(path, str(index)+'.png'))
fold = 'resize__' + str(height) + 'x' + str(width)+ ')'
os.mkdir(fold)
fname = fold
spath = path +'/' +fname
print(spath)
    
for file in files:
    try:
        img = Image.open(os.path.join(path, file))
        img = img.resize((int(height),int(width)),Image.ANTIALIAS)
        img = img.convert('RGB')
        img.save(os.path.join(spath,file))
        print('try')
        print(file)
    except IOError:
        print('except')
        print(file)
        pass