import tensorflow as tf
from matplotlib import pyplot as plt

def load_preprocess(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    input_image = tf.cast(image, tf.float32)
    input_image = tf.image.resize(input_image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = (input_image / 127.5) - 1
    return input_image
  
def getPrediction(filename):

    image = load_preprocess('C:/Users/SPARK/Downloads/flask_files/static/uploads/'+filename)
    image = tf.reshape(image, [1,256,256,3])
    original = tf.reshape(image, [256,256,3])
    original = original* 0.5 + 0.5
    
    model = tf.keras.models.load_model('C:/Users/SPARK/Desktop/Pro/model.h5',compile = False)
    prediction = model(image, training = True)
    prediction = tf.reshape(prediction, [256,256,3])
    prediction = prediction* 0.5 + 0.5
    return (prediction,original)
