import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from numpy import asarray
from PIL import Image, ImageOps
import glob 

images = []

for file_name in glob.glob('my_handwriting/*.jpg'):
    current_image = Image.open(file_name).convert('L')
    images.append(np.asarray(current_image))

images = 1 - (np.array(images) / 255.0)

my_model = tf.keras.models.load_model('my_model')

predictions = my_model.predict(images)

answers = predictions.argmax(axis=1)

with open('answers.txt', 'w') as f:
    for i in answers:
        f.write(str(answers[i]))
        f.write('\n')