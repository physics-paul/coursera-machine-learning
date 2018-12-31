"""*********************************************************

NAME:     ex6

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  1. Support Vector Machines

*********************************************************"""

import numpy as np
from scipy import io
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pdb import set_trace as pb

### Get data ###############################################

mat1 = io.loadmat('ex6data1.mat')
X = mat1['X']
y = mat1['y'].ravel()

mat2 = io.loadmat('ex6data2.mat')
X2 = mat2['X']
y2 = mat2['y'].ravel()

mat3 = io.loadmat('ex6data3.mat')
X3 = mat3['X']
y3 = mat3['y'].ravel()
X3val = mat3['Xval']
y3val = mat3['yval'].ravel()

### 1.1 Example Dataset 1 ##################################

lr1 = SVC(kernel='linear',C=1.0)
lr2 = SVC(kernel='linear',C=100.0)
lr1.fit(X,y)
lr2.fit(X,y)

### output #################################################

fig1 = plt.figure()

plt.scatter(X[(y==0).ravel(),0],X[(y==0).ravel(),1],marker='o')
plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],marker='+')

X_plt = np.linspace(X[:,0].min(),X[:,0].max(),100)
Y_plt1 = -(lr1.intercept_ + lr1.coef_[0,0] * X_plt) / lr1.coef_[0,1]
Y_plt2 = -(lr2.intercept_ + lr2.coef_[0,0] * X_plt) / lr2.coef_[0,1]

plt.plot(X_plt,Y_plt1,color='b',label='C=1')
plt.plot(X_plt,Y_plt2,color='g',label='C=100')
plt.legend()

### 1.2 SVM with Gaussian Kernels #########################

x1 = np.array([1,2,1])
x2 = np.array([0,4,-1])
sigma = 2

def gaussian(x1,x2,sigma):

    return np.exp(-((x1 - x2)**2).sum() / (2 * sigma**2))

gauss = gaussian(x1,x2,sigma)

### 1.2.2 Example Dataset 2 ###############################

lr3 = SVC(kernel='rbf',C=10.0,gamma=20.0)
lr3.fit(X2,y2)

### output ################################################

fig2 = plt.figure()

X_plot,Y_plot = np.meshgrid(*list(map(lambda i : np.arange(X2[:,i].min() - 0.1,X2[:,i].max() + 0.1,0.004),(0,1))))

Z_plot = lr3.predict(np.array([X_plot.ravel(),Y_plot.ravel()]).T)
Z_plot = Z_plot.reshape(X_plot.shape)

plt.contour(X_plot,Y_plot,Z_plot)

plt.scatter(X2[(y2==0).ravel(),0],X2[(y2==0).ravel(),1],marker='o')
plt.scatter(X2[(y2==1).ravel(),0],X2[(y2==1).ravel(),1],marker='+')

plt.show()

### 1.2.3 Example Dataset 3 ###############################

csteps = np.array([0.001,0.01,0.03,0.1,0.3,1,3,10,30,50,100])
gammas = np.array([0.001,0.01,0.03,0.1,0.3,1,3,10,30,50,100])

predict = np.zeros((csteps.shape[0],gammas.shape[0]))

for i,c in enumerate(csteps):
    for j,gam in enumerate(gammas):
        lr = SVC(kernel='rbf',C=c,gamma=gam)
        lr.fit(X3,y3)

        predict[i,j] = (lr.predict(X3val) == y3val.ravel()).sum() / y3val.shape[0]

pos = np.unravel_index(predict.argmax(),predict.shape)

lr = SVC(kernel='rbf',C=csteps[pos[0]],gamma=gammas[pos[1]])
lr.fit(X3,y3)

### output ################################################

print("   ex6 : 1. Support Vector Machines")
print("   The value for C = {1} and the value for sigma = {1}.".format(csteps[pos[0]],gammas[pos[1]]))

fig3 = plt.figure()

X_plot,Y_plot = np.meshgrid(*list(map(lambda i : np.arange(X3[:,i].min() - 0.1,X3[:,i].max() + 0.1,0.004),(0,1))))

Z_plot = lr.predict(np.array([X_plot.ravel(),Y_plot.ravel()]).T)
Z_plot = Z_plot.reshape(X_plot.shape)

plt.contour(X_plot,Y_plot,Z_plot)

plt.scatter(X3[(y3==0).ravel(),0],X3[(y3==0).ravel(),1],marker='o')
plt.scatter(X3[(y3==1).ravel(),0],X3[(y3==1).ravel(),1],marker='+')

plt.show()

###########################################################
