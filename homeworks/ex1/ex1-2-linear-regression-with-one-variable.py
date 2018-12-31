"""*********************************************************

NAME:     ex1

AUTHOR:   Paul Haddon Sanders IV, Ph.D.

VERSION:  2. Linear Regression with one variable

*********************************************************"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pdb import set_trace as pb

### Get data ###############################################

data   = pd.read_csv("ex1data1.txt",header=None)
x_vals = data[0].values
x_vals = x_vals.reshape((x_vals.shape[0],1))
y_vals = data[1]

### 2.2.3 Computing the cost J(theta) ######################

class linear_regression:
    
    def __init__(self,eta=0.01,n_iter=10):
        self.eta    = eta
        self.n_iter = n_iter
        
    def fit(self,x_vals,y_vals):

        self.w0 = np.zeros((1+x_vals.shape[1]))
        
        # self.w0 = np.random.normal(loc=0.0,scale=0.01,size=1+x_vals.shape[1])

        self.cost_ = []

        m = x_vals.shape[0]
        
        for _ in range(self.n_iter):

            output = self.activation(self.input(x_vals))

            errors = y_vals - output

            self.w0[1:] += self.eta * x_vals.T.dot(errors)

            self.w0[0]  += self.eta * errors.sum()

            cost = 0.5/m*(errors**2).sum()
            
            self.cost_.append(cost)

    def input(self,x_vals):

        z = np.dot(x_vals,self.w0[1:]) + self.w0[0]
        
        return z
    
    def activation(self,z):

        ### linear activation ###

        return z

    def predict(self,guess):

        return self.w0[0] + (self.w0[1:] * guess).sum() 
    
### output #################################################
    
a = linear_regression(eta=0.01,n_iter=1)
a.fit(x_vals,y_vals)

costout = round(a.cost_[0],2)

### 2.2.4 Gradient descent #################################

b = linear_regression(eta=0.0001,n_iter=1500)
b.fit(x_vals,y_vals)

### output #################################################

guess1 = round(b.predict(3.5),2)
guess2 = round(b.predict(7.0),2)

x_plot = np.linspace(5,30,1000)
plt.plot(x_plot,b.w0[0] + b.w0[1] * x_plot,'b')
plt.scatter(data.iloc[:,0],data.loc[:,1],marker='x',c='r')
plt.savefig('plot2.png')

print("   ex1 : Problem 2. Linear Regression With One Variable")
print("   cost                   = {}".format(costout))
print("   guess for 35000 people = {}".format(guess1))
print("   guess for 70000 people = {}".format(guess2))
print("   done...")

############################################################
