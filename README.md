# Low-Light-Image-Enhancement

### **Goal of the project:**
To transform low-light images to enhanced/enlightened images using GAN

The images that are captured in low-light conditions, suffer from very poor visibility caused by low contrast, color distortion and significant measurement noise. Our objective is to improve the visibility of low-light images using a deep learning-based method (conditional Generative Adversarial Networks or cGANs).
The basic idea is to train a neural network to generate the illuminated and enhanced version of the low-light image.

### Dataset used
● Low Light paired dataset (LOL): [Google Drive](https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB), [Baidu Pan (Code:acp3)](https://pan.baidu.com/s/1ABMrDjBTeHIJGlOFIeP1IQ)

● Synthetic Image Pairs from Raw Images: [Google Drive](https://drive.google.com/open?id=1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F), [Baidu Pan](https://pan.baidu.com/s/1drsMAkRMlwd9vObAM_9Iog)

We combined two datasets named LOL Dataset (500 images) and Synthetic Image Pairs from Raw Images dataset (1000 images). 
Total of 1450 images were used for training the model and 50 images for test predictions.
Image Resolution - 256 x 256.
We use concatenated images (Lowlight image + Ground Truth) for training network with resolution 256x512.

### Sample input images with groundtruth

Below are the sample images of the dataset that are used to train the network
The input images are concatenated images. LEFT part is the original input image and RIGHT part is the groundTruth of the input image.

![alt sample image1](Images/sample1.png)
![alt sample image1](Images/sample2.png)

### Generator Network - modified UNet

U-Net is an encoder-decoder network where the input is passed through a series of layers that progressively down sample, until a bottleneck layer, at which point the process is reversed. To give the generator a means to circumvent the bottleneck for information like this, we add skip connections, following the general shape of a “U-Net” architecture. Specifically, skip connections between each layer and layer (n−i), where ‘n’ is the total number of layers. Each skip connection simply concatenates all channels at layer ‘i’ with those at layer ‘n−i’.

Input image dimension - 256x256
Output image dimension - 256x256

### Discriminator Network - modified PatchGAN

This discriminator tries to classify if each N×N patch in an image is real or fake. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of D. 

Input image dimension - 256x256
Output image dimension - 30x30

### Model Training

To optimize our networks, we follow the standard approach of alternating between one gradient descent step on Discriminator, then one step on Generator. As suggested in the original GAN paper, rather than training Generator to minimize log (1−D(x, G(x,z)), we instead train to maximize logD(x,G(x,z)). In addition, we divide the objective by 2 while optimizing Discriminator, which slows down the rate at which Discriminator learns relative to Generator. We use mini-batch SGD and apply the Adam optimizer, with a learning rate of 0.0001 for Generator and 0.00005 for Discriminator, and momentum parameters β1= 0.5, β2= 0.999

![alt Training](Images/training.png)

### Fine Tuning hyper-parameters during training

Final Configuration of hyperparameters used for training:
Lambda λ = 115, Generator LR = 0.0001, Discriminator LR = 0.00005, β1= 0.5, β2= 0.999, epochs = 150

![alt sample image1](Images/hyperparameters.png)

### Model Prediction

![alt prediction_1](Images/prediction1.png)
![alt prediction_2](Images/prediction2.png)

These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at generating illuminated or enhanced images from corresponding low-light images.A variety of denoising, deblurring, and enhancement techniques have been proposed including basic image processing techniques, deep learning techniques like CNNs and autoencoders but their effectiveness is limited in extreme conditions. This GAN-based approach seems to perform the best, given current research efforts and may replace many of the traditional image processing methods for illuminating and enhancing low-light images.

### Flask Application deployed on AWS EC2 instance

![alt application_demo](Images/aws_demo.png)
