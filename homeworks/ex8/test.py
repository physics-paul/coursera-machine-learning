class anomaly:

    def __init__(self):
        return self

    def get_multi_gauss(X):

       cov_mat = (X - X.mean(0)).T.dot(X - X.mean(0))

       k = cov_mat.shape[0]
       
       def multi_gauss(x):

           value = np.zeros(x.shape[0])
           
           for i,x in enumerate(X):
               
               val = (x - X.mean(0)).dot(np.linalg.inv(cov_mat))
               val = -0.5*np.diag(val.dot((x - X.mean(0)).T))
               val = np.exp(val)
               val = 1 / np.sqrt((2 * np.pi)**k * np.linalg.det(cov_mat)) * val
               value[i] = val

            return value

       self.multi_gauss = multi_gauss
       
       return self

   
