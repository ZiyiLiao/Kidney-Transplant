import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
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

    def __init__(self,data_dir, sample):
        
        """
        Attributes:
            mean: global mean for rating
            bu  : bias associates with users' rating 
            bi  : bias associates with items' rating
            p   : user-matrix
            q   : item-matrix
        """

        self.data = pd.read_csv(data_dir)
        if sample:
            self.data = self.data[0:1000]
        self.data = self.data.to_numpy()
        self.mean = np.mean(self.data[:,2])
        
        self.user_count = int(max(self.data[:,0]))
        self.item_count = int(max(self.data[:,1]))

        self.bu = None
        self.bi = None
        self.p = None
        self.q = None
        self.h = 0.0000001

    def split(self,test_size, seed = 0):
        train_data, test_data = train_test_split(self.data, test_size = test_size, random_state = seed)
        return train_data, test_data
    
    def fit(self, data, lr = 0.005,reg = 0.4, rank = 10, num_epoch = 10, seed = 0 , elapse = False):
        """
        A method to perform matrix factorization using Stochastic Gradient Descent

        lr        : learning rate, Defalt: 10
        reg       : regularization parameter: lambda, Defalt: 0.4
        rank      : number of latent variables, Default : 10
        num_epoch : number of iteration of the SGD procedure, Default:10
        seed      : seed of calling random state
        """

            
        # initialize user matrix and item matrix
        np.random.seed(seed)
        p = np.random.normal(2.5,1, size = (self.user_count, rank))
        q = np.random.normal(2.5,1, size = (self.item_count, rank))

        # initialize bias
        bu = np.zeros(self.user_count)
        bi = np.zeros(self.item_count)

        start_time = time.time()
        for this_epoch in range(num_epoch):
            
            for row in data:
                u = int(row[0])-1
                i = int(row[1])-1

                # prediction
                pred = self.mean + bu[u] + bi[i] + np.dot(p[u,:], q[i,:])          
                err = row[2] - pred
                    
                # update bias
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])

                # update user and item latent feature matrices
                for f in range(rank):
                    puf = p[u,f]
                    qif = q[i,f]
                    p[u,f]  += lr * (err * qif - reg * puf)
                    q[i,f]  += lr * (err * puf - reg * qif)
        end_time = time.time()
        last = round((end_time - start_time), 4)
        if elapse:
            print("Total time: {}s".format(last))
                    
        # update the instance variariables
        self.bu = bu
        self.bi = bi
        self.p = p
        self.q = q


                    
    def err(self, testset, measure):
        """A method to calculate loss"""

        err = 0
        for row in testset:
            u = int(row[0])-1
            i = int(row[1])-1
            
            if measure == 'rmse':
                # prediction
                err += (row[2] - self.mean - self.bu[u] - self.bi[i]- np.dot(self.p[u,:], self.q[i,:]))**2
                return np.sqrt(err/len(testset))
            if measure == 'mae':
                err += np.abs(row[2] - self.mean - self.bu[u] - self.bi[i]- np.dot(self.p[u,:], self.q[i,:]))
                return err/len(testset)
            return np.sqrt(err/len(testset))

        
    def cv(self, data, measure,lr = 0.005, reg = 0.4, rank = 10, num_epoch = 10, K = 5, verbose = True, plot = False, seed = 0):
        """A method fo perform cross-validation"""

        np.random.seed(seed)
        temp_data = np.random.permutation(data)

        num = len(data)
        fold = num//K
        s = np.append(np.repeat(np.arange(0,K-1),fold) , [4]*(len(data) - (K-1)*fold))

        train_err= [0] * K
        cv_err = [0] * K

        for k in range(K):
            train = temp_data[s!=k,:]
            test = temp_data[s==k,:]
            self.fit(train, lr = lr,reg = reg,rank = rank,num_epoch = num_epoch)
            train_err[k] = self.err(train, measure)
            cv_err[k] = self.err(test, measure)
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
    
    def gridParams(self, measure = ['rmse'], lr = [0.005], reg = [0.4] ,rank = [10], num_epoch = [10], K = [5]):
        params = []
        for mr, l, r, rk, nb, k in itertools.product( measure, lr, reg, rank, num_epoch, K):
            params.append((mr, l, r, rk, nb, k))
        self.params = params

    def tuningParams(self,data, verbose = False, elapse = False):

        loss = []
        for comb in self.params:
            if verbose == True:
                print("stage {}".format(comb))
            err = self.cv(data, measure = comb[0], lr = comb[1], reg = comb[2], rank = comb[3], num_epoch = comb[4], K = comb[5],verbose = verbose)
            loss.append(err)
            print("Min cv err: {}".format(err))
        idx = np.argmin(loss)
        self.best_score = loss[idx]
        self.best_params = self.params[idx]
        self.fit(data,lr = self.best_params[1], reg = self.best_params[2], rank = self.best_params[3], num_epoch = self.best_params[4])        



if __name__ == '__main__':

    func = SGD('../data/ratings.csv',sample = True)
    trainset, testset = func.split(0.25)
    func.gridParams(num_epoch = [10,20], reg = [0.2,0.4], lr = [0.001,0.005], rank = [10,20])
    func.tuningParams(trainset, verbose = True)
    #func.fit(trainset)
    #print(func.err(trainset, measure = 'rmse'))
    print("Best learning rate: {}; Best lambda: {}; Best rank: {}; Best number of epoch: {}".format(func.best_params[1],func.best_params[2], func.best_params[3],func.best_params[4]))


