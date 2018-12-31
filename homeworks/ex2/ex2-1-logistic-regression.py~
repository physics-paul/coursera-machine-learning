"""*********************************************************

NAME:     ex2

AUTHOR:   Paul Haddon Sanders IV, Ph.D.

VERSION:  1. Logistic Regression

*********************************************************"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from pdb import set_trace as pb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

### Get data ###############################################

data   = pd.read_csv("ex2data1.txt",header=None)
x_vals = data.iloc[:,:-1].values
m = x_vals.shape[0]

x_std = np.copy(x_vals)
mean  = [x_vals[:,i].mean() for i in range(x_vals.shape[-1])] 
std   = [x_vals[:,i].std() for i in range(x_vals.shape[-1])]
x_std = (x_vals - mean) / std

y_vals = data.iloc[:,-1].values

### 1.1 Visualizing the data ###############################

plot1 = plt.figure(1)
plt.scatter(*x_vals[y_vals == 0].T,marker='o',c='y',label="Not Admitted")
plt.scatter(*x_vals[y_vals == 1].T,marker='+',c='k',label="Admitted")
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()

### 1.2.1 Warmup exercise: sigmoid function ################

def activation(x_vals,theta):

    z  = np.dot(x_vals,theta[1:]) + theta[0]

    return 1. / (1. + np.exp(-np.clip(z,-250,250)))

### 1.2.2 Cost function and gradient #######################

def cost(theta):

    activate = activation(x_vals,theta)
    
    J = -1 / m * (y_vals.dot(np.log(activate)) + (1-y_vals).dot(np.log(1-activate)))

    return J

def dcost(theta):

    activate = activation(input(x_vals))

    dJ = 1 / m * x_vals.T.dot(activate - y_vals)

    return dJ

theta = np.zeros((1 + x_vals.shape[-1]))
initial_cost = round(cost(theta),3)

### 1.2.3 Learning parameters using fminunc ################

lr = LogisticRegression(solver='lbfgs',multi_class='auto')
lr.fit(x_vals,y_vals)

ideal_theta = np.append(lr.intercept_,lr.coef_)
ideal_cost  = round(cost(ideal_theta),3)

### output #################################################

x_plot = np.linspace(20,100,1000)
y_plot = - (ideal_theta[0] + ideal_theta[1] * x_plot) / ideal_theta[2]
plt.plot(x_plot,y_plot)
plt.ylim([30,100])
plt.xlim([30,100])
plt.show(block=False)

### 1.2.4 Evaluating logistic regression ###################

prediction = round(activation([45,85],ideal_theta),3)

### output #################################################

print("   ex2 : 1. Logistic regression")
print("   initial cost = {}".format(initial_cost))
print("   ideal cost = {}".format(ideal_cost))
print("   admission probability with exam scores (45,85) = {}".format(prediction))

############################################################
