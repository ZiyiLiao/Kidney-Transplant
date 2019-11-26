import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
import time


class SGD:
    """
    The class for performing collaborative filtering with Stochastic Gradient Descent

    Methods for post-processing:
        1. KNN

    Mesures for evaluation:
        1. RMSE: root mean square error
        2. MAE : mean absolute error

    """

    def __init__(self,data_dir, sample = False):
        
        """
        Attributes:
            mean: global mean for rating
            bu  : bias associates with users' rating 
            bi  : bias associates with items' rating
            p   : user-matrix
            q   : item-matrix
        """

        self.data = pd.read_csv(data_dir)
        del self.data['timestamp']
        if sample:
            self.data = self.data[0:1000]
        
        self.mean = np.mean(self.data['rating'])
        self.user_count = len(self.data['userId'].unique())
        self.item_count = len(self.data['movieId'].unique())

        self.bu = None
        self.bi = None
        self.p = None
        self.q = None
        self.item_dict = None

    def split(self,test_size, seed = 0):
        train_data, test_data = train_test_split(self.data, test_size = test_size, random_state = seed)
        return svdData(train_data), svdData(test_data)
    
    def fit(self, data, lr = 0.005,reg = 0.4, rank = 10, num_epoch = 10, seed = 0 , stopping_driv = 0.001, elapse = False):
        """
        A method to perform matrix factorization using Stochastic Gradient Descent

        lr        : learning rate, Defalt: 10
        reg       : regularization parameter: lambda, Defalt: 0.4
        rank      : number of latent variables, Default : 10
        num_epoch : number of iteration of the SGD procedure, Default:10
        seed      : seed of calling random state
        """
        
        tmp1 = self.data['movieId'].unique()
        self.item_dict = dict(zip(tmp1,[i for i in range(self.item_count)]))
            
        # initialize user matrix and item matrix
        np.random.seed(seed)
        p = np.random.normal(2.5,1, size = (self.user_count, rank))
        q = np.random.normal(2.5,1, size = (self.item_count, rank))

        # initialize bias
        bu = np.zeros(self.user_count)
        bi = np.zeros(self.item_count)

        start_time = time.time()
        for this_epoch in range(num_epoch):
            
            for uid, mid, r in data.ratings:
                u = uid - 1
                i = self.item_dict[mid]

                # prediction
                pred = self.mean + bu[u] + bi[i] + np.dot(p[u,:], q[i,:])          
                err = r - pred
                    
                # update bias
                deriv =(err - reg * bu[u])
                if(np.abs(deriv) > stopping_driv):
                    bu[u] += lr * deriv
                deriv = (err - reg * bi[i])
                if(np.abs(deriv) > stopping_driv):
                    bi[i] += lr * deriv

                for f in range(rank):
                    puf = p[u,f]
                    qif = q[i,f]
                    deriv = (err * qif - reg * puf)
                    if(np.abs(deriv) > stopping_driv):
                        p[u,f]  += lr * deriv
                    deriv = (err * puf - reg * qif)
                    if(np.abs(deriv) > stopping_driv):
                        q[i,f]  += lr * deriv
        end_time = time.time()
        last = round((end_time - start_time), 4)
        if elapse:
            print("Total time: {}s".format(last))
                    
        # update the instance variariables
        self.bu = bu
        self.bi = bi
        self.p = p
        self.q = q
        
    def err(self, data, measure = 'rmse'):
        """A method to calculate loss"""

        err = 0

        for uid, mid, r in data.ratings:
            u = uid - 1
            i = self.item_dict[mid]
            
            if measure == 'rmse':
                # prediction
                err += (r- self.mean - self.bu[u] - self.bi[i]- np.dot(self.p[u,:], self.q[i,:]))**2
                return np.sqrt(err/len(data.ratings))
            if measure == 'mae':
                err += np.abs(r - self.mean - self.bu[u] - self.bi[i]- np.dot(self.p[u,:], self.q[i,:]))
                return err/len(data.ratings)

    def KNN(self, testset, K = 5, measure = 'rmse'):
        sim = pairwise_distances(self.p, metric = "cosine")
        err = 0
        for idx, row in testset.iterrows():
            u = int(row['userId']) - 1
            mid = int(row['movieId'])
            i = self.item_dict[mid]
            neighbor = np.argsort(sim[u])[-K-1:-1]
            temp = 0
            
            for k in range(K):
                t = np.dot(self.p[neighbor[k],:], self.q[i,:])
                if t != 0:
                    temp += t


            est_rating = temp/K
            if measure == 'rmse':
                err += (row['rating'] - est_rating) ** 2
                return np.sqrt(err/len(testset))

    def kfold_split(self, data, K = 3,seed = 0):
        kf = KFold(n_splits= K, random_state = seed, shuffle = True)
        trainsets = []
        testsets = []
        self._K = K
        
        for train_index, cv_index in kf.split(data.raw):
            trainset = data.raw.iloc[train_index,:]
            testset = data.raw.iloc[cv_index,:]
            
            trainsets.append(svdData(trainset))
            testsets.append(svdData(testset))
        self._trainsets = trainsets
        self._testsets = testsets
        
    def cv(self, data, measure,lr = 0.005, reg = 0.4, rank = 10, num_epoch = 10, verbose = True, plot = False, seed = 0):
        """A method fo perform cross-validation"""

        train_err= [0] * self._K
        cv_err = [0] * self._K

        for k in range(self._K):
            self.fit(self._trainsets[k], lr = lr,reg = reg,rank = rank,num_epoch = num_epoch)
            train_err[k] = self.err(self._trainsets[k], measure)
            cv_err[k] = self.err(self._testsets[k], measure)
            if verbose:
                print("Train {} : {}   Cross-Validation {} : {}".format(measure, round(train_err[k],4), measure, round(cv_err[k],4)))

        if plot:
            x = np.arange(K) + 1
            plt.title("Train error and cross-validation error")
            plt.plot(x,train_err, label = "train error") 
            plt.plot(x,cv_err, label = "cv error") 
            plt.xlabel("number of folds")
            plt.ylabel("{}".format(measure))
            plt.legend()
            plt.show()
        
        return np.amin(cv_err)
    
    def gridParams(self, lr = [0.005], reg = [0.4] ,rank = [10], num_epoch = [10]):
        params = []
        for l, r, rk, nb in itertools.product(lr, reg, rank, num_epoch):
            params.append((l, r, rk, nb))
        self.params = params

    def tuningParams(self,data, K = 3, measure = 'rmse',verbose = False):
        loss = []
        self.kfold_split(data, K= K)
        for comb in self.params: 
            err = self.cv(data, measure = measure, lr = comb[0], reg = comb[1], rank = comb[2], num_epoch = comb[3], verbose = verbose)
            loss.append(err)
            if verbose:
                print("stage {}".format(comb))
                print("Min cv err: {}".format(err))

        idx = np.argmin(loss)
        self.best_score = loss[idx]
        self.best_params = self.params[idx]



class svdData:

    """
    The class for changing data to a format fit ALS algorithm
    """
    def __init__(self, data, R0):

    # Q : the user-item matrix based on given data
    # R : 1 means the data contains user_u to item_i rating, while 0 means doesn't contain
        self.data = data
        self.R0 = R0
        self.Q, self.R = self._prepare_QR(self.data)

    def __init__(self, data):
        self.raw = data
        self.ratings = [(uid, iid, float(r)) for (uid, iid, r) in data.itertuples(index=False)]
        

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
        train_data = svdData(train_data, self.R0)
        test_data = svdData(test_data, self.R0)
        self.trainsets = [train_data]
        self.testsets = [test_data]
        return train_data , test_data
            
    
    def fit(self, data, rank = 10, reg = 0.1, num_epoch = 10, measure = 'rmse', elapse = False):
        """
        A method to perform matrix factorization

        data      : An svdData format
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
            trainset = svdData(als_data.data.iloc[train_index,:], self.R0)
            cvset = svdData(als_data.data.iloc[cv_index,:], self.R0)
            
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
        