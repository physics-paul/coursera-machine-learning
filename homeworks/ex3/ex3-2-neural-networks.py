"""*********************************************************

NAME:     ex3

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  2. Neural Networks

*********************************************************"""

from sklearn.linear_model import LogisticRegression
from pdb import set_trace as pb
from scipy import io
from PIL import Image as im
import numpy as np

### Get data ###############################################

# There are 5000 training examples.
# Each training example is a 20x20 pixel image of the digit.
# Each pixel is represented by a number indicating
# greyscale intensity.
# This 20x20 grid of pixels is unrolled into a
# 400-dimensional vector.
# Then X is a 5000x400 matrix.
# Y represents the labels for each X training matrix.

mat = io.loadmat('ex3data1.mat')
X   = mat['X']
Y   = mat['y']
Y   = Y.reshape(Y.shape[0])
m   = X.shape[0]

### Get theta network parameters ###########################

# We are given a set of network parameters which have
# already been trained.
# The parameters have dimensions which are sized for a
# neural network with 25 units in the second layer
# and 10 output units.

mat      = io.loadmat('ex3weights.mat')
theta1   = mat['Theta1']
theta2   = mat['Theta2']

### 2.2 Feedforward Propogation and Prediction #############

def activation(X,theta):

    z  = np.dot(X,theta[1:]) + theta[0]

    return 1. / (1. + np.exp(-np.clip(z,-250,250)))

def predict(X,theta1,theta2):

    hidden_layer = np.zeros(theta1.shape[0])
    
    for i in range(hidden_layer.shape[0]):

        hidden_layer[i] = activation(X,theta1[i])

    output_layer = np.zeros(theta2.shape[0])

    for i in range(output_layer.shape[0]):

        output_layer[i] = activation(hidden_layer,theta2[i])

    guess = np.argmax(output_layer) + 1
        
    return guess

guess = np.zeros(X.shape[0])

for i in range(guess.shape[0]):

    guess[i] = predict(X[i],theta1,theta2) 

correct = round(sum(guess == Y) / m * 100,1)

### output #################################################

print("   ex3 : 2. Neural Networks")
print("   Percentage of correct classifications = {}%".format(correct)) 
