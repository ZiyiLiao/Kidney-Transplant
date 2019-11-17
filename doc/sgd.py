import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt

class Collaborative:
    """
    The class for performing collaborative filtering

    Methods for solving matrix factorization: 
        1. Stochastic Gradient Descent
        2. Weighted Alernating Least Square

    Methods for post-processing:
        1. KNN

    Mesures for evaluation:
        1. RMSE: root mean square error

    """

    def __init__(self,data_dir,sample = False):
        
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

    def split(self,test_size, seed = 0):
        test_size = test_size
        np.random.seed(seed)
        temp_data = np.random.permutation(self.data)
        test_idx = int(len(self.data) * test_size)
        test_data = temp_data[0:test_idx]
        train_data = temp_data[test_idx:]
        return train_data, test_data
    
    def MF(self, data, method = 'sgd', lr = 0.005,reg = 0.4, rank = 10, num_batch = 10, seed = 0):
        """
        A function to perform matrix factorization

        method    : method for solving matrix factorization problem
        lr        : learning rate, Defalt: 10
        reg       : regularization parameter: lambda, Defalt: 0.4
        rank      : number of latent variables, Default : 10
        num_batch : number of iteration of the SGD procedure, Default:10
        seed      : seed of calling random state
        """

        # initialize user matrix and item matrix
        np.random.seed(seed)
        p = np.random.normal(0,2.5, size = (self.user_count, rank))
        q = np.random.normal(0,2.5, size = (self.item_count, rank))

        # initialize bias
        bu = np.zeros(self.user_count)
        bi = np.zeros(self.item_count)

        if method == 'sgd':    
            num = len(data)
            batch_size = num//num_batch
            s = np.append(np.repeat(np.arange(0,num_batch-1),batch_size) , [4]*(len(data) - (num_batch-1) * batch_size))
            
            for this_batch in range(num_batch):
                for row in data[s == this_batch,:]:
                    u = int(row[0])-1
                    i = int(row[1])-1

                    # prediction
                    pred = self.mean + bu[u] + bi[i] + np.dot(p[u,:], q[i,:])          
                    err = row[2] - pred
                    
                    # update user and item latent feature matrices
                    for f in range(rank):
                        puf = p[u,f]
                        qif = q[i,f]
                        p[u,f]  += lr * (err * qif - reg * puf)
                        q[i,f]  += lr * (err * puf - reg * qif)

                          
        elif method == 'als':             
            for this_batch in range(num_batch):

                for row in data:
                    u = int(row[0])-1
                    i = int(row[1])-1

                    # prediction
                    pred = self.mean + bu[u] + bi[i] + np.dot(p[u,:], q[i,:])          
                    err = row[2] - pred
                        
                    # update bias
                    nu = np.sum(data[:,0] == u)
                    ni = np.sum(data[:,1] == i)
                    bu[u] += lr * (err - reg * nu * bu[u])
                    bi[i] += lr * (err - reg * ni * bi[i])
                    
                    # update user and item latent feature matrices
                    for f in range(rank):
                        puf = p[u,f]
                        qif = q[i,f]
                        p[u,f]  += lr * (err * qif - reg * nu * puf)
                        q[i,f]  += lr * (err * puf - reg * ni * qif)
        
        self.bu = bu
        self.bi = bi
        self.p = p
        self.q = q

                    
    def err(self, testset, measure):
        """A method to calculate loss"""

        if measure == 'rmse':
            err = 0
            for row in testset:
                u = int(row[0])-1
                i = int(row[1])-1

                # prediction
                err += (row[2] - self.mean - self.bu[u] - self.bi[i]- np.dot(self.p[u,:], self.q[i,:]))**2
            return np.sqrt(err/len(testset))
        
    def cv(self, data,method, measure,lr = 0.005, reg = 0.4, rank = 10, num_batch = 10, K = 5, verbose = True, plot = False, seed = 0):
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
            self.MF(train,method = method,lr = lr,reg = reg,rank = rank,num_batch = num_batch)
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
    
    def gridParams(self, method = ['sgd'], measure = ['rmse'], lr = [0.005], reg = [0.4] ,rank = [10], num_batch = [10], K = [5]):
        params = []
        for md, mr, l, r, rk, nb, k in itertools.product(method, measure, lr, reg, rank, num_batch, K):
            params.append((md, mr, l, r, rk, nb, k))
        self.params = params

    def tuningParams(self,data, verbose = False):

        loss = []
        for comb in self.params:
            if verbose == True:
                print("stage {}".format(comb))
            err = self.cv(data,method = comb[0], measure = comb[1], lr = comb[2], reg = comb[3], rank = comb[4], num_batch = comb[5], K = comb[6],verbose = verbose)
            loss.append(err)
            print("Min cv err: {}".format(err))
        idx = np.argmin(loss)
        self.best_score = loss[idx]
        self.best_params = self.params[idx]
        self.MF(data,method = self.best_params[0],lr = self.best_params[2], reg = self.best_params[3], rank = self.best_params[4], num_batch = self.best_params[5])



if __name__ == '__main__':

    func = Collaborative('ratings.csv',sample = True)
    trainset, testset = func.split(0.2)
    #func.gridParams(method = ['sgd','als'], num_batch = [10,20], reg = [0.2,0.4], lr = [0.001,0.005], rank = [10,20])
    func.gridParams(method = ['sgd','als'], num_batch = [10], reg = [0.4], lr = [0.005,0.002], rank = [10])
    func.tuningParams(trainset, verbose = True)
    print("Best Method: {}; \nBest learning rate: {}; Best lambda: {}; Best rank: {}; Best number of batch: {}".format(func.best_params[0],func.best_params[2],func.best_params[3], func.best_params[4],func.best_params[5]))


