"""*********************************************************

NAME:     ex5

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  2. Learning curves

*********************************************************"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from pdb import set_trace as pb

### Get data ###############################################

# There are 5000 training examples.
# Each training example is a 20x20 pixel image of the digit.
# Each pixel is represented by a number indicating
# greyscale intensity.
# This 20x20 grid of pixels is unrolled into a
# 400-dimensional vector.
# Then X is a 5000x400 matrix.
# Y represents the labels for each X training matrix.

mat1 = io.loadmat('ex5data1.mat')
X    = mat1['X']
Y    = mat1['y'].ravel()
Xval = mat1['Xval']
Yval = mat1['yval'].ravel()
Xtes = mat1['Xtest']
Ytes = mat1['ytest'].ravel()

### 2.1 Learning curves ###################################

class LogisticRegression:
    """Logistic Regression.
    Parameters
    ----------
    eta : float
      Learing rate (between 0.0 and 4.0).
    lambd : float
      Regularization parameter (between 0.0 and 100.0).
    n_iter : int
      Number of passes over the training set.
    
    Attributes
    ----------
    w_ : 1d-array
      Weights of after fitting.
    cost_ : list
      Logistic regression cost function value in each 
      epoch.
    """
    
    def __init__(self,eta=3.0,lambd=0.0,n_iter=10):
        self.eta     = eta
        self.lambd   = lambd
        self.n_iter  = n_iter

    def fit(self, X, Y):
        """Fit training data.
        
        X : {array-like}
          Training vectors.
        Y : {array-like}
          Target values.

        Returns
        -------
        self : object
        """

        self.w_ = np.ones((X.shape[1]+1))
        
        # self.w_ = np.random.normal(loc=0.0,scale=0.1,size=(X.shape[1]+1))
        self.cost_ = [self.cost(X,Y)]

        for i in range(self.n_iter):

            dw_ = self.dcost(X,Y)
            self.w_ += dw_
            self.cost_.append(self.cost(X,Y))
            
    def activation(self,X,theta):
        """Compute linear activation"""

        z = np.dot(X,theta[1:]) + theta[0]

        return z

    def predict(self,X):
        """Return prediction"""
        
        activate = self.activation(X,self.w_)

        return activate
        
    def cost(self,X,Y):
        """Calculate the cost associate with w1 and w2 on the training set"""

        S = X.shape[0]
        
        activate = self.activation(X,self.w_)

        J = 0.5 / S * ((Y - activate)**2).sum() + 0.5 * self.lambd / S * (self.w_[1:]**2).sum()

        return J

    def dcost(self,X,Y):
        """Calculate the updates to the weights using gradient descent"""

        S = X.shape[0]

        activate = self.activation(X,self.w_)

        dw_all = self.eta / S * (Y - activate).dot(X) + self.lambd / S * self.w_[1:]

        dw_bia = self.eta / S * (Y - activate).sum()
        
        dw_ = np.append([dw_bia],dw_all.T)

        return dw_

lr = LogisticRegression(n_iter=1000,eta=0.001,lambd=0.0)
error_train = []
error_val   = []


for i in range(1,X.shape[0]+1):

    lr.fit(X[:i],Y[:i])

    error_train.append(lr.cost(X[:i],Y[:i]))
    error_val.append(lr.cost(Xval,Yval))
    
### output ################################################

X_plt = np.linspace(1,12,12)

plt.plot(X_plt,error_train,label='Train')
plt.plot(X_plt,error_val,label='Cross Validation')
plt.legend()
plt.show()

###########################################################
