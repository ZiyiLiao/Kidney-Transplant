import pandas as pd
import numpy as np
from surprise import SVD
from surprise import accuracy
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV


class Collaborative:

    def __init__(self, data_dir):
        self.df = pd.read_csv(data_dir)
        self.sgd_result = {"measure": None, "best_params": None, "best_score": 0}
        self.als_err = 0
        self.pars = None
        self.test_size = 0.2
        self.train_data = None
        self.test_data = None
        self.n_folds = 3

    def sgd_train(self, test_size, pars, n_folds, measure = "rmse"):

        self.df = self.df[0:10000]
        self.test_size = test_size
        self.pars = pars
        self.n_folds = n_folds
        self.sgd_result['measure'] = measure

        reader = Reader(rating_scale=(1, 5))

        num_test = int(len(self.df)*self.test_size)
        s = np.append([0]*num_test,[1] * (len(self.df) - num_test))

        np.random.seed(2019)
        np.random.shuffle(s)

        train_df = self.df.iloc[s == 1,:]
        test_df = self.df.iloc[s!= 1,:]



        self.train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
        self.test_data = Dataset.load_from_df(test_df[['userId', 'movieId', 'rating']], reader).build_full_trainset().build_testset()

        gs = GridSearchCV(SVD, self.pars, measures = [measure], cv=self.n_folds, refit = True)
        gs.fit(self.train_data)
        cv_rmse = (gs.best_score)

        self.sgd_result['best_params'] = gs.best_params[measure]

        predictions = gs.test(self.test_data)
        self.sgd_result['best_score'] = accuracy.rmse(predictions, verbose = False)

if __name__ == '__main__':

    func = Collaborative('ratings.csv')
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],'reg_all': [0.4, 0.6]}

    print("Stochastic Gradient Descent:")
    func.sgd_train(0.2,param_grid, 3)
    print(func.sgd_result)




