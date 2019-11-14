"""

This is the file to output Absolute Least Square function
Here using PySpark to perform

"""

# import packages

import findspark
findspark.init('/Users/ziyi./spark-2.4.4-bin-hadoop2.7')

import pandas as pd
import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import itertools
from pyspark import SparkContext
from functools import reduce 
from pyspark.sql import DataFrame
from pyspark.conf import SparkConf

# initiate spark
spark = SparkSession.builder.enableHiveSupport().\
    config('spark.executor.memory', '15g'). \
    config('spark.executor.cores','20').  \
    config('spark.executor.instances','20'). \
    config('spark.driver.memory','20g'). \
    getOrCreate()  

# Data preparation

movies = pd.read_csv('movies.csv')
df = pd.read_csv('ratings.csv')
del df['timestamp']


# ratings = spark.createDataFrame(df)
# ratings.show()

print(df.shape)


"""

Calculate Sparsity

"""
numerator = ratings.select("rating").count()
num_users = ratings.select("userId").distinct().count()
num_movies = ratings.select("movieId").distinct().count()
sparsity = (1.0 - (numerator *1.0)/(num_users * num_movies))*100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")



"""
ALS Model

"""


# convert the columns to the proper data types
ratings = ratings.select(ratings.userId.cast("integer"), 
                         ratings.movieId.cast("integer"), 
                         ratings.rating.cast("double"))
# split data
(training_data, test_data) = ratings.randomSplit([0.8, 0.2], seed=2019)

# perform the cross validation to find a set of ideal tuning parameters
def model_fit(r, m, l, k = 5):
    '''
    r: rank -> number of latent features
    m: maximum iterations
    l: lambda -> parameter in the regularization
    k: number of folds
    '''
    
    # model
    als = ALS(userCol = 'userId', itemCol = 'movieId',ratingCol = 'rating',
                rank = r , maxIter = m, regParam = l,
               nonnegative = True, coldStartStrategy = 'drop',
               implicitPrefs = False)
    
    # evaluator
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction") 
    models = {}
    
    # Split training data in to k-fold
    folds = training_data.randomSplit([1/k]*k, seed=2019)
    
    # a function to combine train_data
    def unionAll(dfs):
        return reduce(lambda df1,df2: df1.union(df2.select(df1.columns)), dfs) 
    
    # for k-folds, split the train_data and cv_data
    for i in range(5):
        res = folds[:i]+folds[i+1:]
        train_data = unionAll(res)
        cv_data = folds[i]
        
        # model
        model = als.fit(training_data)
        cv_prediction = model.transform(cv_data)
        
                
        # calculate rmse
        cv_rmse = evaluator.evaluate(cv_prediction)
        models[cv_rmse] = model
    
    min_cv_err = min(models)
    final_model = models[min_cv_err]
    
    return  min_cv_err, final_model


ranks = [5]
maxIters = [5]
regParams = [0.1]

# ranks = [5,10,20]
# maxIters = [5, 50,100]
# regParams = [0.01, 0.05, 0.1]

results = {}
for r, m, l in itertools.product(ranks, maxIters, regParams):
    cv_err, model = model_fit(r,m,l,5)
    params = (r,m,l)
    results[cv_err] = (model, params)

err = min(results)
best_model = results[err][0]
best_params = results[err][1]
rank = best_params[0]
itr = best_params[1]
lmda = best_params[2]


print("**best model**")
print("rank: ", rank, " max iterations: ", itr, " lambda: ", lmda)
print("cross validation: ", err)


