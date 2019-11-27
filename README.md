# Movies Recommender System

This is a reommender system building with movies rating [Movielens](http://grouplens.org/datasets/movielens/)
The whole system segments users as **New User** and **Exsiting User** and apply different analysis on each segment.

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


----------
