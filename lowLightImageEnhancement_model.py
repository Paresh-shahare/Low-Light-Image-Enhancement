'''
Authors: Paresh Shahare
Topic: Low Light Image Enhancement using conditional Generative Adversarial Network
Framework: Tensorflow 2.4.1

File: lowLightImageEnhancement_model.py
'''

# Import libraries
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt

'''
Mount the drive and Load dataset
In random jittering, the image is resized to 286 x 286 and then randomly cropped to 256 x 256
In random mirroring, the image is randomly flipped horizontally i.e left to right.
'''

from google.colab import drive
drive.mount('/content/drive')

PATH = r'/content/drive/MyDrive/Final_Project_cdac/BrightningTrain/new_concat_1500'
print('Path:',PATH)

print(os.listdir(PATH+'/new_concat-1500-train'))

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


# Loading Dataset

def load(image_file):
  image = tf.io.read_file(image_file)					# Reading image using Tensorflow function
  image = tf.image.decode_png(image)					# Decoding the image

  w = tf.shape(image)[1]

  w = w // 2
  input_image = image[:, :w, :]							# Splitting the concatenated image into input image and GroundTruth
  real_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)		# Casting into float32 format
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image




inp, re = load(PATH+'/new_concat-1500-train/2.png')


# Uncomment below code to plot sample image

'''
plt.figure()
plt.title('input-lowlight')
plt.imshow(inp/255.0)									# casting to int for matplotlib to show the image
plt.figure()
plt.title('real-enhanced')
plt.imshow(re/255.0)
print(inp.shape)
'''

# Data Pre-processing

def resize(input_image, real_image, height, width):		# Resize the image to 256x256
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image):				# Randomly crop the resized image 
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]




def normalize(input_image, real_image):					# normalizing the images to [-1, 1]
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image



'''
Random jittering

Resize an image to bigger height and width
Randomly crop to the target size
Randomly flip the image horizontally
'''



@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)
  print(input_image.shape)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)
  print('crop',input_image.shape)
  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image


# Uncomment below code to plot sample input image
'''
plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = random_jitter(inp, re)
  plt.subplot(2, 2, i+1)
  plt.imshow(rj_inp/255.0)
  plt.axis('off')
plt.show()
'''




# Create train and test dataset


def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image




train_dataset = tf.data.Dataset.list_files(PATH+'/new_concat-1500-train/*.png')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


test_dataset = tf.data.Dataset.list_files(PATH+'/new_concat-1500-test/*.png')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)






# GAN Generator Network
'''
The architecture of generator is a modified U-Net.
Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
There are skip connections between the encoder and decoder (as in U-Net).
'''


OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_batchnorm=True):   	# Downsampling operation in encoder consist of CONV -> BatchNorm -> LeakyReLU
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)


def upsample(filters, size, apply_dropout=False):		# Upsampling operation in decoder consist of TransposeConv -> BatchNorm -> Dropout
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)



def Generator():										# Generator model
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [										# Encoder 
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [											# Decoder
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []											# Skip connections between encoder and decoder
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])		# Concatenation using skip connections

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()									# Create generator model



# Uncomment below code to plot sample image using generator
'''
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)


gen_output = generator(inp[tf.newaxis,...], training=True)
plt.imshow(gen_output[0,...])
'''



'''
Generator loss
It is a sigmoid cross entropy loss of the generated images and an array of ones.
This also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
This allows the generated image to become structurally similar to the target image.
The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100.
'''


LAMBDA = 115


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss





'''
Build the Discriminator
The Discriminator is a PatchGAN.
Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
The shape of the output after the last layer is (batch_size, 30, 30, 1)
Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
Discriminator receives 2 inputs.
Input image and the target image, which it should classify as real.
Input image and the generated image (output of generator), which it should classify as fake.
We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
'''



def Discriminator():									# Discriminator model 
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()


# Uncomment below code to plot sample image using discriminator and plot discriminator model
'''
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
'''



'''
Discriminator loss

The discriminator loss function takes 2 inputs; real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since these are the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since these are the fake images)
Then the total_loss is the sum of real_loss and the generated_loss
'''

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  	# BinaryCrossentropy loss


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss




'''
Hyper-parameters for training
Optimizer : Adam Optimizer
Parameters:
lambda: 115
Generator LR: 0.0001
Discriminator LR: 0.00005
Momentum parameters: beta_1:0.5 beta_2:0.999
'''

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)		# Adam optimizer 
discriminator_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5)



def generate_images(model, test_input, tar):				# To visualize predicted images during model training
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

# Uncomment below code to plot early sample predictions on test images
'''
for example_input, example_target in test_dataset.take(1):	# Generate predictions on test images
  generate_images(generator, example_input, example_target)
'''




EPOCHS = 150												# No. of epochs




@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)								# Generating output from generator model

    disc_real_output = discriminator([input_image, target], training=True)			# Calculating Discriminator real output
    disc_generated_output = discriminator([input_image, gen_output], training=True) # Calculating Discriminator generated output

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)	# Calculating generator loss
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)									# Calculating discriminator loss

  generator_gradients = gen_tape.gradient(gen_total_loss,							# Calculating generator gradient
                                          generator.trainable_variables)			# Calculating discriminator gradient
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,						# Applying gradients using optimizer
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  
  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss




# This is where actual training takes place

def fit(train_ds, epochs, test_ds):
  g_t_loss = []
  g_g_loss = []
  g_l1_loss = []
  d_loss = []
  
  for epoch in range(epochs):
    start = time.time()

    #display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch+1)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      if (n % 10) == 0:
        print('.', end='')
      #if (n+1) % 100 == 0:
        #print()
      gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, epoch)
      gen_total_loss_last = float(gen_total_loss)
      gen_gan_loss_last = float(gen_gan_loss) 
      gen_l1_loss_last = float(gen_l1_loss)
      disc_loss_last = float(disc_loss)
    print()
    print(f"gen_total_loss : {gen_total_loss_last}, gen_gan_loss : {gen_gan_loss_last}, gen_l1_loss : {gen_l1_loss_last}, disc_loss : {disc_loss_last}")

    g_t_loss.append(gen_total_loss_last)
    g_g_loss.append(gen_gan_loss_last)
    g_l1_loss.append(gen_l1_loss_last)
    d_loss.append(disc_loss_last)


    if (epoch + 1) % 100 == 0:							# Saving model at epoch =100
      save_model(generator,filepath)
    if (epoch + 1) % 150 == 0:							# Saving model at epoch =150
      save_model(generator,filepath)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)

  return g_t_loss, g_g_loss, g_l1_loss, d_loss



# Train the model
g_t_loss, g_g_loss, g_l1_loss, d_loss = fit(train_dataset, EPOCHS, test_dataset)


# Save the model in .h5 format
generator.save("/content/drive/MyDrive/Projects_chkp/model_chpts_100_1500BrTrDS_lambda130/model115_150.h5")

