"""*********************************************************

NAME:     ex7

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  2. Principal Component Analysis

*********************************************************"""

import numpy as np
from scipy import io
from PIL import Image as im
import matplotlib.pyplot as plt
from pdb import set_trace as pb

### Get data ###############################################

mat1  = io.loadmat('ex7data1.mat')
mat2  = io.loadmat('ex7faces.mat')
mat3  = np.loadtxt('kmeans.txt')

X     = mat1['X']
X_std = (X - X.mean(0)) / X.std(0)

A = mat2['X'] + 256.0/2.0
A_std = (A - A.mean(0)) / A.std(0)

C_std = mat3 / 255.0

### 2.2 Implementing PCA ###################################

class PCA:

    def __init__(self,k=3):
        self.k = k

    def reduction(self,X):
        
        covariance = X.T.dot(X) / X.shape[0]

        self.eigvalues,self.eigvectors = np.linalg.eig(covariance)

        k_vectors = self.eigvalues.argsort()[::-1][:self.k]

        Xbasis    = np.linalg.inv(self.eigvectors).dot(X.T).T

        self.Xreduc    = Xbasis[:,k_vectors]

        eigreduce = self.eigvectors[:,k_vectors]

        self.Xrecov    = eigreduce.dot(self.Xreduc.T).T

        return self
        
### 2.3.2 Reconstructing and approximation of the data #####

pca1 = PCA(k=1)
pca1.reduction(X_std)

### 2.4.1 PCA on Faces #####################################

pca2 = PCA(k=100)
pca2.reduction(A_std)

### 2.5 PCA for visualization ##############################

pca3 = PCA(k=2)
pca3.reduction(C_std)

### output #################################################

print('   ex7 : 2.2 Principal Component Analysis')
print('   First eigenvector : {}'.format(str(pca1.eigvectors[1])))
print('   First dimension reduction example : {}'.format(str(round(pca1.Xreduc[0,0],3))))
print('   First recovery example : {}'.format(str(pca1.Xrecov[0])))

fig1 = plt.figure()

X_plt  = np.array([[a * eig for a in np.linspace(-2,0,100)] for eig in pca1.eigvectors])
X_plt = np.array([xval * X.std(0) + X.mean(0) for xval in X_plt])

plt.scatter(X[:,0],X[:,1])
plt.plot(X_plt[0,:,0],X_plt[0,:,1])
plt.plot(X_plt[1,:,0],X_plt[1,:,1])

fig2 = plt.figure()

plt.scatter(X_std[:,0],X_std[:,1])
plt.scatter(pca1.Xrecov[:,0],pca1.Xrecov[:,1])
plt.show()

all_im = im.new('L', (32*20,32*10))

pca2.Xrecov = pca2.Xrecov * A.std(0) + A.mean(0)

images1 = np.reshape(A[:100],(10,10,1024))
images2 = np.reshape(pca2.Xrecov[:100],(10,10,1024))

for i,row in enumerate(images1):
    for j,ima in enumerate(row):

        ima = np.reshape(ima,(32,32)).T
        ima = im.fromarray(ima)

        all_im.paste(ima,(i*32,j*32))

for i,row in enumerate(images2):
    for j,ima in enumerate(row):

        ima = np.reshape(ima,(32,32)).T
        ima = im.fromarray(ima)

        all_im.paste(ima,((i+10)*32,j*32))
        
all_im.show()

fig3 = plt.figure()

plt.scatter(-pca3.Xreduc[:,0],pca3.Xreduc[:,1])
plt.show()

############################################################
