"""*********************************************************

NAME:     ex8

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  1. Anomaly detection

*********************************************************"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from pdb import set_trace as pb

### Get data ###############################################

mat1  = io.loadmat('ex8data1.mat')
mat2  = io.loadmat('ex8data2.mat')

X = mat1['X']
Xval = mat1['Xval']
yval = mat1['yval'].ravel()

X2    = mat2['X']
Xval2 = mat2['Xval']
yval2 = mat2['yval'].ravel()

### 1.1 Gaussian distribution ##############################

class anomaly:

    def get_multi_gauss(self, X):

       cov_mat = (X - X.mean(0)).T.dot(X - X.mean(0)) / X.shape[0]

       k = cov_mat.shape[0]
       
       def multi_gauss(x):
           
           value = np.zeros(x.shape[0])

           cov_mat_inv = np.linalg.inv(cov_mat)

           cov_mat_det = np.linalg.det(cov_mat)
           
           for i,xval in enumerate(x):
               
               val = np.exp(-0.5*(xval - X.mean(0)).dot(cov_mat_inv).dot(xval - X.mean(0)))
               val = 1 / np.sqrt((2 * np.pi)**k * cov_mat_det) * val
               value[i] = val           

           return value
 
       self.multi_gauss = multi_gauss

    def get_epsilon(self,X,Xval,yval):

        predict = self.multi_gauss(Xval)

        minE = predict.min()
        maxE = predict.max()

        epsilon = np.linspace(minE,maxE,1000)

        f1 = np.zeros(epsilon.shape[0])

        for i,eps in enumerate(epsilon):

            tp = ((predict <= eps) * (yval == 1)).sum()
            fp = ((predict <= eps) * (yval == 0)).sum()
            fn = ((predict >  eps) * (yval == 1)).sum()

            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)

            f1[i] = 2 * prec * rec / (prec + rec)

        self.epsilon_min = epsilon[f1.argmax()]

        predict = self.multi_gauss(X)
        
        self.outliers    = X[predict <= self.epsilon_min]
        self.nonoutliers = X[predict >  self.epsilon_min]

### 1.2 Estimating parameters for a Gaussian ###############

ano = anomaly()
ano.get_multi_gauss(X)

### output #################################################

Xplt = np.linspace(0,30,100)
Yplt = np.linspace(0,30,100)

Xplt,Yplt = np.meshgrid(Xplt,Yplt)
Zplt = np.array([Xplt.ravel(),Yplt.ravel()]).T
Zplt = np.resize(ano.multi_gauss(Zplt),Xplt.shape)

plt.contour(Xplt,Yplt,Zplt)
plt.scatter(X[:,0],X[:,1],marker='x')
plt.show()

### 1.3 Selecting the threshold, epsilon ###################

ano.get_epsilon(X,Xval,yval)

### output #################################################

print('   ex8 : 1. Anomaly detection')
print('   First example dataset epsilon = {}'.format(ano.epsilon_min))

Xplt = np.linspace(0,30,100)
Yplt = np.linspace(0,30,100)

Xplt,Yplt = np.meshgrid(Xplt,Yplt)
Zplt = np.array([Xplt.ravel(),Yplt.ravel()]).T
Zplt = np.resize(ano.multi_gauss(Zplt),Xplt.shape)
 
plt.contour(Xplt,Yplt,Zplt)
plt.scatter(ano.outliers[:,0],ano.outliers[:,1],marker='o')
plt.scatter(ano.nonoutliers[:,0],ano.nonoutliers[:,1],marker='x')
plt.show()

### 1.4 High dimensional dataset ###########################

ano = anomaly()
ano.get_multi_gauss(X2)
ano.get_epsilon(X2,Xval2,yval2)

### output #################################################

print('   Second example dataset epsilon = {}'.format(ano.epsilon_min))
print('   Second example anomalies found = {}'.format(ano.outliers.shape[0]))

############################################################
