'''
Authors: Paresh Shahare
Topic: Low Light Image Enhancement using conditional Generative Adversarial Network
Framework: Tensorflow 2.4.1

File: test.py
'''


# Load libraries
from tensorflow.keras.models import Sequential, save_model, load_model

# Load model
model = tf.keras.models.load_model("/content/drive/MyDrive/Projects_chkp/model_chpts_100_1500BrTrDS_lambda130/model115_150.h5")


def load_preprocess(image_file):			# Pre-processing the input image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    input_image = tf.cast(image, tf.float32)
    input_image = tf.image.resize(input_image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = (input_image / 127.5) - 1
    return input_image
  
def getPrediction(filename):				# Loading the model and processing the test image to get the enlightened output image
    image = load_preprocess(filename)
    image = tf.reshape(image, [1,256,256,3])
    model = tf.keras.models.load_model(r'/home/wireshark/Documents/CDAC/Project/trained_models/model_100.h5',compile = False)
    prediction = model(image, training = True)
    prediction = tf.reshape(prediction, [256,256,3])
    plt.figure()
    plt.imshow(prediction * 0.5 + 0.5)
    plt.axis('off')
    plt.show()
    return True

filename = r'/home/wireshark/Documents/CDAC/Project/dataset/test/low/test_image.png'	# Test image path
getPrediction(filename)