import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from pdb import set_trace as pb

mat1 = io.loadmat('ex8_movies.mat')
mat2 = io.loadmat('ex8_movieParams.mat')

Y = mat1['Y']
R = mat1['R']
X = mat2['X']
T = mat2['Theta']

pb()

class collaborative_cost:
    """Collaborative filtering cost function.

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
    x_ : 2d-array, shape = [n_movies,n_features]
      Attributes for each movie, where n_movies is the number of movies, and n_features is the number of features.
    w_ : 2d-array, shape = [n_users,n_features]
      Attributes for each movie, where n_users is the number of users, and n_features is the number of features.
    cost_ : list
      Linear regression cost function value in each epoc.

    """

    print("")
    print("   COLLABORATIVE FILTERING ALGORITHM INITIALIZED")
    print("   CREATED BY : PAUL SANDERS")
    print("")
    
    def __init__(self,eta=3.0,lambd=0.0,n_iter=1000):
        self.eta     = eta
        self.lambd   = lambd
        self.n_iter  = n_iter

    def fit(self, X, Y):
        """Fit training data.
        
        X : {array-like}, shape = [S,M]
          Training vectors, where S is the number of samples 
          and M is the number of features.
        Y : {array-like}, shape = [S,K]
          Target values, where S is the number of samples
          and K is the number of classifications.

        Returns
        -------
        self : object

        """

        print('   Please be patient. The cost is being minimized.')
        print('')
        
        S = X.shape[0]
        M = X.shape[1]
        K = Y.shape[1]

        self.w1_ = np.random.normal(loc=0.0,scale=0.7,size=(self.n_nodes,M+1))
        self.w2_ = np.random.normal(loc=0.0,scale=0.7,size=(K,self.n_nodes+1))

        self.cost_ = []

        for i in range(self.n_iter):

            dw1_,dw2_ = self.dcost(X,Y)
            self.w1_ += dw1_
            self.w2_ += dw2_
            self.cost_.append(self.cost(X,Y))
            print('   cost = {}'.format(round(self.cost_[-1],7)),end="\r")

        print('')
        print('\n   Minimization Complete.')
        
    def activation(self,X,theta):
        """Compute logistic sigmoid activation"""

        z = np.dot(X,theta[1:]) + theta[0]

        return 1. / (1. + np.exp(-np.clip(z,-250,250)))

    def predict(self,X):
        """Return class labels after hidden layer evaluation"""

        S = X.shape[0]
        
        hidden_layer = np.zeros((S,self.w1_.shape[0]))
        output_layer = np.zeros((S,self.w2_.shape[0]))

        for i in range(self.w1_.shape[0]):

            hidden_layer[:,i] = self.activation(X,self.w1_[i])   

        for i in range(self.w2_.shape[0]):

            output_layer[:,i] = self.activation(hidden_layer,self.w2_[i])

        return hidden_layer,output_layer

    def cost(self,X,Y):
        """Calculate the cost associate with w1 and w2 on the training set"""

        S = X.shape[0]

        activate = self.predict(X)[1]

        J = -1 / S * (Y * np.log(activate)           \
            + (1-Y) * np.log(1-activate)).sum( )     \
            + 0.5 * self.lambd / S * ((self.w1_[:,1:]**2).sum() \
            + (self.w2_[:,1:]**2).sum())

        return J

    def dcost(self,X,Y):
        """Calculate the updates to the weights using gradient descent"""

        S = X.shape[0]

        a2,a3 = self.predict(X)

        """Calculate w2 weights"""

        delta3 = Y - a3
        Delta2_all = delta3.T.dot(a2)
        Delta2_bias = delta3.T.sum(1)
        Delta2 = np.append([Delta2_bias],Delta2_all.T,axis=0).T

        dw2_ = self.eta * (1 / S * Delta2 + self.lambd / S * self.w2_)

        ###Calculate w1 weights"""

        delta2      = delta3.dot(self.w2_[:,1:]) * a2 * (1 - a2)

        Delta1_all  = delta2.T.dot(X)
        Delta1_bias = delta2.T.sum(1)
        Delta1      = np.append([Delta1_bias],Delta1_all.T,axis=0).T

        dw1_ = self.eta * (1 / S * Delta1 + self.lambd / S * self.w1_)

        return dw1_,dw2_

