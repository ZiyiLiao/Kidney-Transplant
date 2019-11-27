# Movies Recommender System

This is a reommender system building with [Movielens](http://grouplens.org/datasets/movielens/) movie ratings.The whole system segments users as **New User** and **Exsiting User** and apply different analysis on each segment inspired by [Scikit-Learn](https://scikit-learn.org/stable/)

## New User

## Exsiting User
For exsiting user, we used collaborative filtering analysis to get the users' protential preferences.

The preprocessing process is the process performing matrix factorization.In this project, I used two methods to apply Sigular Value Decomposition:

 + Stochastic Gradient Descent
 + Alternating Least Squares

The measures to do the evaluation is :
 + Root Mean Square Error
 + Mean Absolute Error
However, in the methods comparison, I mainly used RMSE

The prediction process is to predict the rating after performing matrix facotrization and the results can be used to recommend movies. Here I used KNN algorithm to do so.

**Benchmark**
```
Training time:

 SVD : 72.5138s
 ALS : 21.6902s

Evaluating RMSE:

  Testing Error
SVD 0.018424 
ALS 1.3086509715004353
SVD-KNN 0.0035207775636804306
ALS-KNN 0.010562332691041293

```
