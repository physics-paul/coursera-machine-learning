"""*********************************************************

NAME:     ex4

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  2. Backpropagation

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

mat = io.loadmat('ex4data1.mat')
X   = mat['X']
Y   = mat['y']
Y   = Y.reshape(Y.shape[0])

### Reshape Y ##############################################

n_class_labels = np.unique(Y).shape[0]
Y_exp = np.zeros((Y.shape[0],n_class_labels))

for i,val in enumerate(Y):

    # we need to change the Y outputs into an array, since
    # our classifier is with many different options, not
    # only one digit.
    
    index = val - 1
    
    Y_exp[i][index] = 1

Y = np.copy(Y_exp)
    
### Sizes of matrices ######################################

# X      = [S,M]
# theta1 = [N,M+1]
# a2     = [S,N]
# theta2 = [K,N+1]
# a3     = [S,K]
# Y      = [S,K]

S   = X.shape[0]
M   = X.shape[1]
K   = n_class_labels

# S = total number of training examples
# M = total number of features, excluding the bias unit
# N = total number of units in the hidding layer, excluding
#     the bias unit
# K = total number of output classifications

### 2.3 Backpropagation ####################################

def activation(X,theta):

    z  = np.dot(X,theta[1:]) + theta[0]

    return 1. / (1. + np.exp(-np.clip(z,-250,250)))

def predict(X,theta1,theta2):
    
    hidden_layer = np.zeros((X.shape[0],theta1.shape[0]))
    output_layer = np.zeros((X.shape[0],theta2.shape[0]))

    for i in range(theta1.shape[0]):

        hidden_layer[:,i] = activation(X,theta1[i])   

    for i in range(theta2.shape[0]):

        output_layer[:,i] = activation(hidden_layer,theta2[i])
    
    return hidden_layer,output_layer

def cost_regularized(X,Y,theta1,theta2,lambd):

    S = X.shape[0]
    
    activate = predict(X,theta1,theta2)[1]

    J = -1 / S * (Y * np.log(activate) + (1-Y) * np.log(1-activate)).sum() + 0.5 * lambd / S * ((theta1**2).sum() + (theta2**2).sum())

    return J

###

theta1 = np.random.normal(loc=0.0,scale=2.0,size=(25,M+1))
theta2 = np.random.normal(loc=0.0,scale=2.0,size=(K,25+1))

costs = []

cost_reg = round(cost_regularized(X,Y,theta1,theta2,0),6)

costs.append(cost_reg)

for _ in range(20):
    
    a2,a3 = predict(X,theta1,theta2)

    ###

    delta3      = Y - a3

    Delta2_all  = delta3.T.dot(a2)
    Delta2_bias = delta3.T.sum(1)
    Delta2      = np.append([Delta2_bias],Delta2_all.T,axis=0).T

    dtheta2     = 1 / S * Delta2

    ###

    delta2      = delta3.dot(theta2[:,1:]) * a2 * (1 - a2)

    Delta1_all  = delta2.T.dot(X)
    Delta1_bias = delta2.T.sum(1)
    Delta1      = np.append([Delta1_bias],Delta1_all.T,axis=0).T

    dtheta1     = 1 / S * Delta1

    theta1 = theta1 + dtheta1
    theta2 = theta2 + dtheta2
    
    cost_reg = round(cost_regularized(X,Y,theta1,theta2,0),6)

    costs.append(cost_reg)

pb()
    
### 2.5 Regularized Neural Networks ########################

lambd = 1

theta1 = np.random.normal(loc=0.0,scale=0.01,size=(25,M+1))
theta2 = np.random.normal(loc=0.0,scale=0.01,size=(K,25+1))

costs = []

cost_reg = round(cost_regularized(X,Y,theta1,theta2,1),6)

costs.append(cost_reg)

for _ in range(10):
    
    a2,a3 = predict(X,theta1,theta2)

    ###

    delta3      = Y - a3

    Delta2_all  = delta3.T.dot(a2)
    Delta2_bias = delta3.T.sum(1)
    Delta2      = np.append([Delta2_bias],Delta2_all.T,axis=0).T

    dtheta2     = 1 / S * Delta2 + lambd / S * theta2

    ###

    delta2      = delta3.dot(theta2[:,1:]) * a2 * (1 - a2)

    Delta1_all  = delta2.T.dot(X)
    Delta1_bias = delta2.T.sum(1)
    Delta1      = np.append([Delta1_bias],Delta1_all.T,axis=0).T

    dtheta1     = 1 / S * Delta1 + lambd / S * theta1

    theta1 = theta1 + dtheta1
    theta2 = theta2 + dtheta2
    
    cost_reg = round(cost_regularized(X,Y,theta1,theta2,1),6)

    costs.append(cost_reg)
