import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
import time


class mfData:

    """
    The class for changing data to a format fit matrix factorization
    """
    def __init__(self, data, R0):

    # Q : the user-item matrix based on given data
    # R : selector matrix,  1 means the data contains user_u to item_i rating, while 0 means doesn't contain
        self.data = data
        self.R0 = R0
        self.ratings = [(uid, iid, float(r)) for (uid, iid, r) in data.itertuples(index=False)]
        self.Q, self.R = self._prepare_QR(self.data)

        

    def _prepare_QR(self, data):
        temp = self.R0.copy()
        for u, i, r in self.ratings:
            temp[i][u] = r

        Q = temp.values
        R = (Q > 0) *1
        return Q, R


class MF:
    """
    The class for performing matrix factorization with Singular Vector Deviation


    Methods for Singular Vector Deviation:
        1. Stochastic Gradient Descent
        2. Alternating Least Squares

    Measures for evaluation:
        1. RMSE: root mean square error
        2. MAE : mean absolute error

    Method for Prediction:
        1. K-nearest Neighbors

    """

    def __init__(self,data_dir, sample = False):
        
        """
        Attributes:
            mean: global mean for rating
            _bu : bias associates with users' rating 
            _bi : bias associates with items' rating
            p   : user-matrix
            q   : item-matrix

            user_count : number of users
            item_count : number of items
            item_dict  : a dictionary for looking up item index in the item-matrix

            data : user - item matrix in DataFrame format
            Q    : user - item matrix in numpy array format
            R0   : origin selector matrix, all elements are 0

            _K : K-folds
            error: training error

            params: parameters for tuning
            best_params: parameters with minimum cross-validation error

        """

        self.df = pd.read_csv(data_dir)
        del self.df['timestamp']
        
        self.mean = np.mean(self.df['rating'])
        self.user_count = len(self.df['userId'].unique())
        self.item_count = len(self.df['movieId'].unique())
        self.data = (self.df.pivot(index='userId', columns='movieId', values='rating')).fillna(0)

        self._bu = None
        self._bi = None
        self.p = None
        self.q = None


        self.item_dict = None

        self.Q = self.data.values
        self.R0 = self.data * 0

        

        self._K = None
        self.error = None
        self._trainsets = None
        self._testsets = None
        self.params = None
        self.best_params = None



    def split(self,test_size, seed = 0):
        train_data, test_data = train_test_split(self.df, test_size = test_size, random_state = seed)
        train_data = mfData(train_data, self.R0)
        test_data = mfData(test_data, self.R0)
        self._trainsets = [train_data]
        self._testsets = [test_data]
        return train_data , test_data
    
    def sgd(self, data, lr = 0.005,reg = 0.4, rank = 10, num_epoch = 10, seed = 0 , stopping_driv = 0.001, measure = 'rmse', elapse = False):
        """
        A method to perform matrix factorization using Stochastic Gradient Descent

        lr        : learning rate, Defalt: 10
        reg       : regularization parameter: lambda, Defalt: 0.4
        rank      : number of latent variables, Default : 10
        num_epoch : number of iteration of the SGD procedure, Default:10
        seed      : seed of calling random state
        """
        
        # store the algorim status
        self._algo = 'sgd'
        
        tmp1 = self.df['movieId'].unique()
        self.item_dict = dict(zip(tmp1,[i for i in range(self.item_count)]))
            
        # initialize user matrix and item matrix
        np.random.seed(seed)
        p = np.random.normal(2.5,1, size = (self.user_count, rank))
        q = np.random.normal(2.5,1, size = (self.item_count, rank))

        # initialize bias
        bu = np.zeros(self.user_count)
        bi = np.zeros(self.item_count)

        # count the training time
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

                # updata user-matrix and item-matrix
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

        # store the training error
        self.error = self.err(data, measure)
        
 
        
    def als(self, data, rank = 10, reg = 0.1, num_epoch = 10, measure = 'rmse', elapse = False):
        """
        A method to perform matrix factorization using Alternating Least Square

        data      : An mfData format
        reg       : regularization parameter: lambda, Defalt: 0.4
        rank      : number of latent variables, Default : 10
        num_epoch : number of iteration of the SGD procedure, Default:10
        measure   : evaluation method
        elapse    : if true, print the time of fitting 

        """
        
        # store the algorithm status
        self._algo = 'als'
        
        # set up the user-matrix and item-matrix
        I = np.eye(rank)
        np.random.seed(0)
        p = np.random.normal(2.5,1, size = (self.Q.shape[0], rank))
        q = np.random.normal(2.5,1, size = (self.Q.shape[1],rank))

        start_time = time.time()
        for this_epoch in range(num_epoch):

            # update user-matrix
            for u, Iu in enumerate(data.R):
                j = np.nonzero(Iu)[0]
                nu = sum(Iu)
                if nu != 0:
                    A = q[j,:].T.dot(q[j,:]) + nu * reg * I
                    r = data.Q[u][data.Q[u]!=0]
                    V = q[j,:].T.dot(r.T)
                    p[u,:] = np.dot(np.linalg.inv(A), V)
            
            # update item-matrix
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

        # update instance variables
        self.q = q
        self.p = p

        # update training error
        self.error = self.err(data, measure)
        
    def err(self, data, measure = 'rmse'):
        """A method to calculate loss"""
        
        if self._algo == 'sgd':
            err = 0
            for uid, mid, r in data.ratings:
                u = uid - 1
                i = self.item_dict[mid]

                if measure == 'rmse':
                    # prediction
                    err += (r- self.mean - self.bu[u] - self.bi[i]- np.dot(self.p[u,:], self.q[i,:]))**2
                    return np.sqrt(err/len(data.ratings))
                if measure == 'mae':                
                    return err/len(data.ratings)

        if self._algo == 'als':
            if measure == 'rmse':
                loss = np.sum((data.R * (self.Q - np.dot(self.p, self.q.T))) ** 2)/ np.sum(data.R)
                return np.sqrt(loss)
            if measure == 'mae':
                loss = np.sum(np.abs(data.R * (self.Q - np.dot(self.p, self.q.T))))/ np.sum(data.R) 
                return loss


    def kfold_split(self, svd_data, K = 3,seed = 0):
        kf = KFold(n_splits= K, random_state = seed, shuffle = True)
        trainsets = []
        testsets = []
        # update _K
        self._K = K
        
        for train_index, test_index in kf.split(svd_data.data):
            trainset = svd_data.data.iloc[train_index,:]
            testset = svd_data.data.iloc[test_index,:]
            # convert to mfData and then store into list
            trainsets.append(mfData(trainset,self.R0))
            testsets.append(mfData(testset,self.R0))
        self._trainsets = trainsets
        self._testsets = testsets
        
    def cv(self, data, measure,lr = 0.005, reg = 0.4, rank = 10, num_epoch = 10, verbose = True, plot = False, seed = 0):
        """A method fo perform cross-validation"""
    
        train_err= [0] * self._K
        test_err = [0] * self._K

        for k in range(self._K):
            
            if self._algo == 'sgd':
                self.sgd(self._trainsets[k], lr = lr,reg = reg,rank = rank,num_epoch = num_epoch)

            if self._algo == 'als':
                self.als(self._trainsets[k], rank = rank, num_epoch = num_epoch, measure = measure)

            train_err[k] = self.error
            test_err[k] = self.err(self._testsets[k], measure)
            if verbose:
                print("Train {} : {}   Cross-Validation {} : {}".format(measure, round(train_err[k],4), measure, round(test_err[k],4)))
            
        if plot:
            x = np.arange(K) + 1
            plt.title("Train error and cross-validation error")
            plt.plot(x,train_err, label = "train error") 
            plt.plot(x,test_err, label = "cv error") 
            plt.xlabel("number of folds")
            plt.ylabel("{}".format(measure))
            plt.legend()
            plt.show()
        
        return np.amin(test_err)
    
    def gridParams(self, algo, lr = [0.005], reg = [0.4] ,rank = [10], num_epoch = [10]):
        params = []
        
        # set up the algorithm
        self._algo = algo

        if self._algo == 'als':
            for rk, r, np in itertools.product(rank, reg, num_epoch):
                params.append((rk, r, np))
        else:
            for l, r, rk, np in itertools.product(lr, reg, rank, num_epoch):
                params.append((l, r, rk, np))

        self.params = params

    def tuningParams(self,data, K = 3, measure = 'rmse',verbose = False):
        loss = []

        # split into K-folds
        self.kfold_split(data, K= K)

        for comb in self.params: 

            if self._algo == 'sgd':
                err = self.cv(data, measure = measure, lr = comb[0], reg = comb[1], rank = comb[2], num_epoch = comb[3], verbose = verbose)
            
            elif self._algo == 'als':
                err = self.cv(data, measure = measure, reg = comb[1], rank = comb[0], num_epoch = comb[2], verbose = verbose)
            loss.append(err)
            if verbose:
                print("stage {}".format(comb))
                print("Min cv err: {}".format(err))
        # store the minimum cross-validation error and the corresponding parameters
        idx = np.argmin(loss)
        self.best_score = loss[idx]
        self.best_params = self.params[idx]
        
    def KNN(self, train, test, K = 5, measure = 'rmse', elapse = False):
        """A method to predict using item-item recommendation"""

        # calculate the item-similarity matrix
        sim = pairwise_distances(self.q, metric = "cosine")

        err = 0
        start_time = time.time()
        for uid, mid, r in test.ratings:
            u = uid - 1
            i = self.item_dict[mid]

            # sort the idx with the similarity
            neighbors_idx = np.argsort(sim[i])

            # delete neighbors with rating 0
            neighbor_rating = np.trim_zeros(train.Q[u,neighbors_idx])[0:K]

            # predict
            est = sum(neighbor_rating)/np.count_nonzero(neighbor_rating)
            if measure == 'rmse':
                err += (r - est) **2
                return np.sqrt(err/len(test.ratings))
            elif measure == 'mae':
                err += np.abs(r - est)
                return err/len(test.ratings)
        end_time = time.time()
        last = end_time - start_time
        if elapse:
            print("KNN predict time: {}s".format(last))
    