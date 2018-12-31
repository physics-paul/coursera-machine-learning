"""*********************************************************

NAME:     ex5

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  3. Polynomial Regression

*********************************************************"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from pdb import set_trace as pb

### Get data ###############################################

class data:

    def __init__(self,x,m=8,y=False):
        self.d   = x
        self.mn  = x.mean(0)
        self.st  = x.std(0)
        self.m   = m
        self.n   = self.norm(x.mean(0),x.std(0))
        self.o   = self.orig(x.mean(0),x.std(0))
        if y:
            self.p    = self.n
        else:
            self.p    = self.poly(x.mean(0),x.std(0))
        
    def norm(self,mn,st):
        
        n = (self.d - mn) / st
        
        return n

    def orig(self,mn,st):
        
        o = self.d * st + mn

        return o

    def poly(self,mn,st,normalize=True):

        if normalize:
            d = self.norm(mn,st)
        else:
            d = self.d
            
        xa = np.copy(d)

        for i in range(self.m-1):

            xa = np.append(xa,d**(i+2),axis=1)

        return xa
        
mat1 = io.loadmat('ex5data1.mat')
X    = data(mat1['X'])
Y    = data(mat1['y'].ravel(),y=True)
Xval = data(mat1['Xval'])
Yval = data(mat1['yval'].ravel(),y=True)
Xtes = data(mat1['Xtest'])
Ytes = data(mat1['ytest'].ravel(),y=True)

### 3.1 Learning Polynomial Regression ####################

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
    
    def __init__(self,lambd=0.0,n_iter=10):
        self.lambd   = lambd
        self.n_iter  = n_iter

    def fit(self, X, Y, eta):
        """Fit training data.
        
        X : {array-like}
          Training vectors.
        Y : {array-like}
          Target values.
        eta : float
          Learing rate (between 0.0 and 4.0).

        Returns
        -------
        self : object
        """

        self.w_ = np.zeros((X.shape[1]+1))
        self.eta = eta
        
        # self.w_ = np.random.normal(loc=0.0,scale=0.1,size=(X.shape[1]+1))
        self.cost_ = [self.cost(X,Y)]

        for i in range(self.n_iter):

            dw_ = self.dcost(X,Y)
            self.w_ += dw_
            self.cost_.append(self.cost(X,Y))

            if i > 3 and self.cost_[-3] < self.cost_[-1]:
                break
            
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

        dw_all = self.eta * (1 / S * (Y - activate).dot(X) + self.lambd / S * self.w_[1:])

        dw_bia = self.eta / S * (Y - activate).sum()
        
        dw_ = np.append([dw_bia],dw_all.T)

        return dw_

lr1 = LogisticRegression(n_iter=3000,lambd=0.0)

error_train0 = []
error_val0   = []

for i in range(1,X.p.shape[0]+1):

    lr1.fit(X.p[:i],Y.n[:i],eta=0.001 + 0.0005*i)
    
    error_train0.append(lr1.cost(X.p[:i],Y.n[:i]))
    error_val0.append(lr1.cost(Xval.p,Yval.n))

### 3.2 Adjusting the regularization parameter ############

lr2 = LogisticRegression(n_iter=2000,lambd=1.0)
lr3 = LogisticRegression(n_iter=2000,lambd=100.0)

error_train1 = []
error_val1   = []

for i in range(1,X.p.shape[0]+1):

    lr2.fit(X.p[:i],Y.n[:i],eta=0.001 + 0.0005*i)
    
    error_train1.append(lr2.cost(X.p[:i],Y.n[:i]))
    error_val1.append(lr2.cost(Xval.p,Yval.n))

lr3.fit(X.p,Y.n,eta=0.001)

### 3.3 Selecting lambda using a cross validation set #####

error_train2 = []
error_val2 = []

for i in range(11):

    lr4 = LogisticRegression(n_iter=3000,lambd=i)
    lr4.fit(X.p,Y.p,eta=0.001)
    error_train2.append(lr4.cost(X.p,Y.p))
    error_val2.append(lr4.cost(Xval.p,Yval.n))

### 3.4 Computing test set error ##########################

lr5 = LogisticRegression(n_iter=3000,lambd=3.0)
lr5.fit(X.p,Y.p,eta=0.001)
cost = round(lr5.cost(Xtes.p,Ytes.n),5)

### 3.5 Plotting learning curves with randomly selected exa

lr6 = LogisticRegression(n_iter=2000,lambd=0.01)

error_train3 = []
error_val3   = []

X_choice = np.random.choice(range(X.p.shape[0]),12)
Y_choice = np.random.choice(range(Y.n.shape[0]),12)

Xval_choice = np.random.choice(range(Xval.p.shape[0]),12)
Yval_choice = np.random.choice(range(Yval.n.shape[0]),12)

X1 = X.p[X_choice]
Y1 = Y.n[Y_choice]

Xval1 = Xval.p[Xval_choice] 
Yval1 = Yval.n[Yval_choice]

for i in range(1,X1.shape[0]+1):

    lr6.fit(X1[:i],Y1[:i],eta=0.001)
    
    error_train3.append(lr6.cost(X1[:i],Y1[:i]))
    error_val3.append(lr6.cost(Xval1[:i],Yval1[:i]))

### output ################################################

print('   ex5 : 3. Polynomial regression')
print('   Cost over test set with lambda = 3 : {}'.format(cost))

fig1 = plt.figure()

X_plt = np.linspace(-65,65,100)
X_plt = np.reshape(X_plt,(X_plt.shape[0],1))
X_plt = data(X_plt)
Y_plt = data(lr1.predict(X_plt.poly(X.mn,X.st)),y=True)

plt.scatter(X.d,Y.d,marker='x',color='r')
plt.plot(X_plt.d,Y_plt.orig(Y.mn,Y.st))
plt.xlim([-80,80])
plt.ylim([-60,40])
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression (lam = 0)')

fig2 = plt.figure()

X_plt = np.linspace(1,12,12)

plt.plot(X_plt,error_train0,label='Train')
plt.plot(X_plt,error_val0,label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.ylim([0,20])
plt.legend()
plt.title('Polynomial Regression Learning Curve (lam = 0)')

fig3 = plt.figure()

X_plt = np.linspace(-60,60,100)
X_plt = np.reshape(X_plt,(X_plt.shape[0],1))
X_plt = data(X_plt)
Y_plt = data(lr2.predict(X_plt.poly(X.mn,X.st)),y=True)

plt.scatter(X.d,Y.d,marker='x',color='r')
plt.plot(X_plt.d,Y_plt.orig(Y.mn,Y.st))
plt.xlim([-80,80])
plt.ylim([-60,40])
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression (lam = 1)')

fig4 = plt.figure()

X_plt = np.linspace(1,12,12)

plt.plot(X_plt,error_train1,label='Train')
plt.plot(X_plt,error_val1,label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.ylim([0,20])
plt.legend()
plt.title('Polynomial Regression Learning Curve (lam = 1)')

fig5 = plt.figure()

X_plt = np.linspace(-60,60,100)
X_plt = np.reshape(X_plt,(X_plt.shape[0],1))
X_plt = data(X_plt)
Y_plt = data(lr3.predict(X_plt.poly(X.mn,X.st)),y=True)

plt.scatter(X.d,Y.d,marker='x',color='r')
plt.plot(X_plt.d,Y_plt.orig(Y.mn,Y.st))
plt.xlim([-80,80])
plt.ylim([-10,40])
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression (lam = 100)')

fig6 = plt.figure()

X_plt = np.linspace(0,10,11)

plt.plot(X_plt,error_train2,label='Train')
plt.plot(X_plt,error_val2,label='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.ylim([0,0.25])
plt.legend()
plt.title('Polynomial Regression Learning Curve for lambda')

plt.show()

fig7 = plt.figure()

plt.scatter(Xtes.d,Ytes.d,marker='x',color='r')

X_plt = np.linspace(-60,60,100)
X_plt = np.reshape(X_plt,(X_plt.shape[0],1))
X_plt = data(X_plt)
Y_plt = data(lr5.predict(X_plt.poly(X.mn,X.st)),y=True)

plt.plot(X_plt.d,Y_plt.orig(Y.mn,Y.st))
plt.xlim([-80,80])
plt.ylim([-60,80])
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression test set (lambda = 3)')

fig8 = plt.figure()

X_plt = np.linspace(1,12,12)

plt.plot(X_plt,error_train3,label='Train')
plt.plot(X_plt,error_val3,label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.ylim([0,20])
plt.legend()
plt.title('Polynomial Regression Learning Curve (lam = 0.1)')

plt.show()

############################################################
