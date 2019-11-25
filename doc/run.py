import ALS
from SGD import SGD

func = SGD('../data/ratings.csv',sample = True)
trainset, testset = func.split(0.25)
#func.gridParams(num_epoch = [10,20], reg = [0.2,0.4], lr = [0.001,0.005], rank = [10,20])
#func.tuningParams(trainset, verbose = True)
func.fit(trainset, elapse = True)
print(func.err(trainset, measure = 'rmse'))
#print("Best learning rate: {}; Best lambda: {}; Best rank: {}; Best number of epoch: {}".format(func.best_params[1],func.best_params[2], func.best_params[3],func.best_params[4]))
