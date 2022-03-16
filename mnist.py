import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from numpy import asarray

# FUNCTIONS COPIED FROM: 
# https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/multi-class_classification_with_MNIST.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=multiclass_tf2-colab&hl=en#scrollTo=pedD5GhlDC-y
 
def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    
    plt.legend()

def create_model(my_learning_rate):
  """Create and compile a deep neural net."""
  
  # All models in this course are sequential.
  model = tf.keras.models.Sequential()

  # The features are stored in a two-dimensional 28X28 array. 
  # Flatten that two-dimensional array into a a one-dimensional 
  # 784-element array.
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

  # Define the first hidden layer.   
  model.add(tf.keras.layers.Dense(units=32, activation='relu'))
  
  # Define a dropout regularization layer. 
  model.add(tf.keras.layers.Dropout(rate=0.2))

  # Define the output layer. The units parameter is set to 10 because
  # the model must choose among 10 possible output values (representing
  # the digits from 0 to 9, inclusive).
  #
  # Don't change this layer.
  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))     
                           
  # Construct the layers into a model that TensorFlow can execute.  
  # Notice that the loss function for multi-class classification
  # is different than the loss function for binary classification.  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  
  return model    

def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
  """Train the model by feeding it data."""

  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, 
                      validation_split=validation_split)
 
  # To track the progression of training, gather a snapshot
  # of the model's metrics at each epoch. 
  epochs = history.epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist    

# IMPORT THE MNIST DATASET
(training_features, training_label),(test_features, test_labels) = tf.keras.datasets.mnist.load_data()

# NORMALIZE THE VALUES IN EACH IMAGE
normalized_training = training_features / 255.0 
normalized_test = test_features / 255.0

learning_rate = .003
epochs = 100 
batch_size = 4000 
validation_split = .2

my_model = create_model(learning_rate)
epochs, hist = train_model(my_model, normalized_training, training_label, 
                           epochs, batch_size, validation_split)

metrics = ['accuracy']

print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=test_features,y=test_labels, batch_size=batch_size)

print("\nSaving the model...")
my_model.save('my_model')


