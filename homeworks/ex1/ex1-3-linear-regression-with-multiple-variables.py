"""*********************************************************

NAME:     ex1

AUTHOR:   Paul Haddon Sanders IV, Ph.D.

VERSION:  3. Linear Regression with multiple variables

*********************************************************"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pdb import set_trace as pb

### Get data ###############################################

data   = pd.read_csv("ex1data2.txt",header=None)
x_vals = data.iloc[:,:-1].values
y_vals = data.iloc[:,-1].values

### 3.1. Feature Normalization #############################

mean  = np.array([x_vals[:,i].mean() for i in range(x_vals.shape[-1])])
std   = np.array([x_vals[:,i].std() for i in range(x_vals.shape[-1])])
x_std = (x_vals - mean) / std

### 3.2. Gradient Descent ##################################

class linear_regression:
    
    def __init__(self,eta=0.01,n_iter=10):
        self.eta    = eta
        self.n_iter = n_iter
        
    def fit(self,x_vals,y_vals):

        self.w0 = np.random.normal(loc=0.0,scale=0.01,size=1+x_vals.shape[1])

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
    
a = linear_regression(eta=0.01,n_iter=50)
a.fit(x_std,y_vals)

### 3.2.1 Optional (ungraded) exercise: Selecting learning r

learn_rate = 0.003

b = linear_regression(eta=learn_rate,n_iter=150)
b.fit(x_std,y_vals)

guess = np.array([1650,3])
guess_std = (guess - mean) / std

prediction = round(b.predict(guess_std))

### output #################################################

x_plot = np.linspace(1,150,150)
plt.plot(x_plot,b.cost_,c='g')
plt.ylabel('Cost J')
plt.xlabel('Number of Iterations')
plt.xlim([0,50])
plt.ylim([0,7*(10**10)])
plt.show(block=False)

print('   ex1 : 3. Linear Regression with Multiple Variables')
print('   prediction for 1650 sq. ft. and 3 bedrooms = {}'.format(prediction))

### 3.3. Normal Equations ##################################

# append 1's to the x_values

x_vals_tot  = np.zeros((x_std.shape[0],1+x_std.shape[1]))
x_vals_tot[:,1:] = x_std
x_vals_tot[:,0]  = 1

# theta = (x.T x)^-1 x.T y

xTx = x_vals_tot.T.dot(x_vals_tot)
w0_act = np.linalg.inv(xTx).dot(x_vals_tot.T).dot(y_vals)

guess_std = np.append([1],guess_std)
prediction = round(w0_act.dot(guess_std),0)

### output #################################################

print('   prediction for 1650 sq. ft. and 3 bedrooms, actual = {}'.format(prediction))

############################################################
