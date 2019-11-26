import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import itertools
import time


class alsData:

    """
    The class for changing data to a format fit ALS algorithm
    """
    def __init__(self, data, R0):

    # Q : the user-item matrix based on given data
    # R : 1 means the data contains user_u to item_i rating, while 0 means doesn't contain
        self.data = data
        self.R0 = R0
        self.Q, self.R = self._prepare_QR(self.data)
        

    def _prepare_QR(self, data):
        temp = self.R0.copy()
        for idx, row in data.iterrows():
            u = int(row['userId'])
            i = int(row['movieId'])
            temp[i][u] = row['rating']

        Q = temp.values
        R = (Q > 0) *1
        return Q, R



class ALS:
    """
    The class for performing collaborative filtering with Alternating Least Squares

    Mesures for evaluation:
        1. RMSE: root mean square error
        2. MAE : mean absolute error

    """
    
    def __init__(self, data_dir, sample = False):
        self.df = pd.read_csv(data_dir)
        del self.df['timestamp']
        if sample:
            self.df = self.df[0:1000]
        self.data = (self.df.pivot(index='userId', columns='movieId', values='rating')).fillna(0)
        # user-item matrix
        self.Q = self.data.values
        self.R0 = self.data * 0
        # user matrix
        self.p = None
        # item matrix
        self.q = None
        # kfolds
        self.K = None
        self.error = None
        self.trainsets = None
        self.testsets = None
        self.params = None
        self.best_params = None
        
        
                    
    def split(self,test_size = 0.25, seed = 0):
        """A method for train_test_split"""
        train_data, test_data = train_test_split(self.df,test_size = test_size, random_state = seed)
        train_data = alsData(train_data, self.R0)
        test_data = alsData(test_data, self.R0)
        self.trainsets = [train_data]
        self.testsets = [test_data]
        return train_data , test_data
            
    
    def fit(self, data, rank = 10, reg = 0.1, num_epoch = 10, measure = 'rmse', elapse = False):
        """
        A method to perform matrix factorization

        data      : An alsData format
        reg       : regularization parameter: lambda, Defalt: 0.4
        rank      : number of latent variables, Default : 10
        num_epoch : number of iteration of the SGD procedure, Default:10
        measure   : evaluation method
        elapse    : if true, print the time of fitting 

        """
        I = np.eye(rank)
        np.random.seed(0)
        p = np.random.normal(2.5,1, size = (self.Q.shape[0], rank))
        q = np.random.normal(2.5,1, size = (self.Q.shape[1],rank))

        start_time = time.time()
        for this_epoch in range(num_epoch):

            for u, Iu in enumerate(data.R):
                j = np.nonzero(Iu)[0]
                nu = sum(Iu)
                if nu != 0:
                    A = q[j,:].T.dot(q[j,:]) + nu * reg * I
                    r = data.Q[u][data.Q[u]!=0]
                    V = q[j,:].T.dot(r.T)
                    p[u,:] = np.dot(np.linalg.inv(A), V)

            for i, Ii in enumerate(data.R.T):
                j = np.nonzero(Ii)[0]
                ni = sum(Ii)
                if ni != 0:
                    A = p[j,:].T.dot(p[j,:]) + ni * reg * I
                    r = data.Q.T[i][data.Q.T[i]!=0]
                    V = p[j,:].T.dot(r) 
                    q[i,:] = np.dot(np.linalg.inv(A), V)
        end_time = time.time()
        last = round((end_time - start_time), 4)
        if elapse:
             print("Total time: {}s".format(last))
        self.q = q
        self.p = p
        self.error = self.err(data)
        
        
    def err(self, data, measure = 'rmse'):
        """data: als data"""
        if measure == 'rmse':
            loss = np.sum((data.R * (self.Q - np.dot(self.p, self.q.T))) ** 2)/ np.sum(data.R)
            return np.sqrt(loss)
        if measure == 'mae':
            loss = np.sum(np.abs(data.R * (self.Q - np.dot(self.p, self.q.T))))/ np.sum(data.R) 
            return loss
    
    def kfolds_split(self, als_data, K = 3, seed = 0):
        kf = KFold(n_splits= K, random_state = seed, shuffle = True)
        self.K = K
        trainsets = []
        cvsets = []
        
        for train_index, cv_index in kf.split(als_data.data):
            trainset = alsData(als_data.data.iloc[train_index,:], self.R0)
            cvset = alsData(als_data.data.iloc[cv_index,:], self.R0)
            
            trainsets.append(trainset)
            cvsets.append(cvset)
        self.trainsets = trainsets
        self.cvsets = cvsets
        
    def cv(self,rank = 10, reg = 0.1, num_epoch = 10, measure = 'rmse', verbose = True, plot = False):
        train_loss = []
        test_loss = []
        for k in range(self.K):
            train = self.trainsets[k]
            test = self.cvsets[k]
            self.fit(train, rank = rank, num_epoch = num_epoch, measure = measure)
            train_loss.append(self.error)
            test_loss.append(self.err(test,measure = measure))
            
            if verbose:
                print("Train {} : {}   Cross-Validation {} : {}".format(measure, train_loss[k], measure,test_loss[k]))
        
        if plot:
            x = np.arange(self.K) + 1
            plt.title("Train error and cross-validation error")
            plt.plot(x,train_loss, label = "train error") 
            plt.plot(x,test_loss, label = "test error") 
            plt.xlabel("number of folds")
            plt.ylabel("{}".format(measure))
            plt.legend()
            plt.show()
        
        return np.min(test_loss)
    
    def gridParams(self, rank = [10],  reg = [0.1] , num_epoch = [10], K = [3]):
        params = []
        for rk, r, np, k in itertools.product(rank, reg, num_epoch, K):
            params.append((rk, r, np, k))
        self.params = params
    
    def tuningParams(self, data, measure = 'rmse', verbose = True):
        self.kfolds_split(data)
        loss = []

        for comb in self.params:
            test_err = self.cv(measure = measure, rank = comb[0], reg = comb[1], num_epoch = comb[2],verbose = verbose)
            loss.append(test_err)
            if verbose == True:
                print("stage {}".format(comb))
                print("Min cv err: {}".format(test_err))

        idx = np.argmin(loss)
        self.best_score = loss[idx]
        self.best_params = self.params[idx]
        
        #self.fit(data, rank = self.best_params[1],reg = self.best_params[2],  num_epoch = self.best_params[3])        
        