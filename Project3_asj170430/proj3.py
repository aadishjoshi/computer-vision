# based on code from https://www.tensorflow.org/tutorials
# python proj3.py big/x_train.csv big/y_train.csv big/x_test.csv big/y_test.csv

import os
import sys
import tensorflow as tf
import numpy as np

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# specify path to training data and testing data
def read_file(filename):
	x_train_2d = np.loadtxt(filename, dtype="uint8", delimiter=",")
	x_train_3d = x_train_2d.reshape(-1,28,28,1)
	x_train = x_train_3d
	return x_train

# define the training model
def define_model():
	model = tf.keras.models.Sequential([
	    tf.keras.layers.MaxPool2D(4, 4, input_shape=(28,28,1)),
	    tf.keras.layers.Conv2D(7, (3,3), padding='same', activation=tf.nn.relu),
	    tf.keras.layers.MaxPool2D(2, 2),
	    tf.keras.layers.Flatten(),
	    tf.keras.layers.Dense(512, activation=tf.nn.relu),
	    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])
	return model

def compile(model):
	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

def train(model, x_train,y_train):
	print("train")
	model.fit(x_train, y_train, epochs=50)

def test(model, x_test, y_test):
	print("evaluate")
	model.evaluate(x_test, y_test)

if __name__ == "__main__":
	if len(sys.argv) != 4 and len(sys.argv) != 5 :
	  print(sys.argv[0], "takes 4 or 5 arguments. Not ", len(sys.argv)-1)
	  print("Arguments: x_train y_train Example: ",
	        sys.argv[0]," x_train.csv y_train.csv x_test.csv y_test.csv")
	  sys.exit()

	x_train = sys.argv[1]
	y_train = sys.argv[2]
	x_test = sys.argv[3]
	y_test = sys.argv[4]

	x_train = read_file(x_train)
	y_train = np.loadtxt(y_train, dtype="uint8", delimiter=",")

	print("Pre processing x of training data")
	x_train = x_train / 255.0

	model = define_model()
	compile(model)
	train(model, x_train,y_train)

	x_test = read_file(x_test)
	y_test = np.loadtxt(y_test, dtype="uint8", delimiter=",")

	#x_test, y_test = getTestData()

	print("Reading testing data")
	x_test = x_test / 255.0

	scores = model.evaluate(x_test,y_test)
	print("Loss: %.2f%%" % (scores[0]*100))
	print("Accuracy: %.2f%%" % (scores[1]*100))
	#test(model, x_test, y_test)