"""*********************************************************

NAME:     ex7

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  1. K-means Clustering

*********************************************************"""

import numpy as np
from scipy import io
from PIL import Image as im
from pdb import set_trace as pb

### Get data ###############################################

mat1  = io.loadmat('ex7data2.mat')
mat2  = io.loadmat('bird_small.mat')
mat3  = im.open('image.jpg')
X     = mat1['X']
A     = mat2['A']
B     = np.asarray(mat3)

Aproc     = np.resize(A,(A.shape[0]**2,A.shape[2]))

np.savetxt('kmeans.txt',Aproc)

Bproc     = np.resize(B,(B.shape[0]**2,B.shape[2]))

### 1.1 Implementing K-means ###############################

class kmeans:

    def __init__(self,k=3,n_iters=10):
        self.k       = k
        self.n_iters = n_iters

    def find_centroid(self,X):
        """Find centroid using K-means algorithm

        Parameters
        ----------
        X : {array-like}, shape = [n_samples,n_features]
          Training vectors, where n_samples is the number of samples and n_features is the number of features

        Returns
        -------
        self : object
        
        """

        final_distance = np.zeros(5)
        new_centroid   = np.zeros((5,self.k,X.shape[1]))
        
        """ Run the algorithm randomly several times """
        
        for iter1 in range(5):
        
            """ Randomly select centroids from (unique) X
                values """
            
            add_choice = X[np.random.choice(X.shape[0],self.k,replace=False)]

            choice = np.unique(add_choice,axis=0)
            
            while len(choice) < self.k:
                
                add_choice = X[np.random.choice(X.shape[0],self.k,replace=False)]
                
                choice = np.append(choice,add_choice,axis=0)
                choice = np.unique(choice,axis=0)
                
            new_centroid[iter1] = choice[:self.k]
            
            """ Proceed to find the new centroid """
            
            for iter2 in range(self.n_iters):

                """ Calculate distances """
                
                distances = np.array([((X - val)**2.0).sum(1) for val in new_centroid[iter1]])

                """ Find the nearest centroid position """
                
                centroid_number = distances.T.argmin(1)

                """ Get the new centroids by averaging """
                
                new_centroid[iter1] = np.array([X[np.where(centroid_number == val)].sum(0) / np.where(centroid_number == val)[0].shape[0] for val in range(self.k)])

            """ Check the error in the new centroid """
                
            distances = np.array([((X - val)**2.0).sum(1) for val in new_centroid[iter1]])
            centroid_number = distances.T.argmin(1)
            final_distance[iter1] = np.array([((X[np.where(centroid_number == i)] - val)**2.0).sum() for i,val in enumerate(new_centroid[iter1])]).sum()

        """ Select the centroid with the lowest error """
            
        self.centroid = new_centroid[final_distance.argmin()]
        
        return self

    def compress_data(self,X):

        distances = np.array([((X - val)**2.0).sum(1) for val in self.centroid])
        
        Xcopy = self.centroid[distances.T.argmin(1)]

        return Xcopy

    def change_pixel(self,pixel):
        
        distances = np.array([((pixel - val)**2.0).sum() for val in self.centroid])

        self.centroid[distances.T.argmin()] = pixel

        return self
        
### 1.4 Image compression with K-means #####################

km1 = kmeans(k=16)
km1.find_centroid(Aproc)

Acomp = km1.compress_data(Aproc)
Acomp = Acomp.astype('uint8')
Acomp = np.resize(Acomp,A.shape)

### 1.5 Use your own image #################################

km2  = kmeans(k=10)
km2.find_centroid(Aproc)
km2.change_pixel([255,255,255])

Bcomp     = km2.compress_data(Bproc)
Bcomp     = np.resize(Bcomp,B.shape)
Bcomp     = Bcomp.astype('uint8')

### output #################################################

image1 = im.new('RGB',(A.shape[0]*2,A.shape[0]))
image1_orig = im.fromarray(A)
image1_comp = im.fromarray(Acomp)

image1.paste(image1_orig,(0,0))
image1.paste(image1_comp,(Acomp.shape[0],0))

image1.show()

image2 = im.new('RGB',(B.shape[0]*2,B.shape[0]))
image2_orig = im.fromarray(B)
image2_comp = im.fromarray(Bcomp)

image2.paste(image2_orig,(0,0))
image2.paste(image2_comp,(Bcomp.shape[0],0))

image2.show()
