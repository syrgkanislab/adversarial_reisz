rm(list=ls())

setwd("~/Documents/research/minimax_riesz")

library("kernlab")
library("MASS")
source('kernel_fast.R')
source('sim.R')
source('minimax_aux.R')
source('primitives.R')
source('stage1.R')
library("quantreg")


########################################
# MSE of minimax Riesz vs Lasso, Dantzig
########################################

# generate data
n=2000
p=10
set.seed(2)
data=get_sim(n,p)
Y=data[[1]]
D=data[[2]]
X=data[[3]]
pi=data[[4]]
alpha_true=data[[5]]
hist(pi)

# split into two folds, train on one, validate on other
folds <- split(sample(n, n,replace=FALSE), as.factor(1:2))

# train - obs
Y.1=as.matrix(Y[folds[[1]]])
D.1=as.matrix(D[folds[[1]]])
X.1=as.matrix(X[folds[[1]],])
pi.1=as.matrix(pi[folds[[1]],])
alpha_true.1=as.matrix(alpha_true[folds[[1]],])

# test - obs
Y.2=as.matrix(Y[folds[[2]]])
D.2=as.matrix(D[folds[[2]]])
X.2=as.matrix(X[folds[[2]],])
pi.2=as.matrix(pi[folds[[2]],])
alpha_true.2=as.matrix(alpha_true[folds[[2]],])

### minimax

lambda=5/n^(0.4)
mu=10^(-10)

# precompute SS
ss.1=get_ss(D.1,X.1)
ss.2=get_ss(D.2,X.2)
ss_eval=get_ss_eval(D.1,D.2,X.1,X.2)

# train - estimator
Delta.1=get_Delta(ss.1,lambda)
Omega.1=get_Omega(ss.1,Delta.1,lambda)
K_WW.1=ss.1[[1]] # eval
alpha_hat.1=get_alpha(ss.1,Delta.1,Omega.1,K_WW.1,lambda,mu)

# train - oracle mse
oracle_mse.1=get_oracle_mse(D.1,pi.1,alpha_true.1,alpha_hat.1)
#oracle_mse.1[[1]]
oracle_mse.1[[2]]

# test - estimator
K_WW.2=ss_eval[[1]] # train on 1, eval on 2
alpha_hat.2=get_alpha(ss.1,Delta.1,Omega.1,K_WW.2,lambda,mu) # train on 1, eval on 2

# test - oracle mse
oracle_mse.2=get_oracle_mse(D.2,pi.2,alpha_true.2,alpha_hat.2)
#oracle_mse.2[[1]]
oracle_mse.2[[2]]

### Lasso

dict=b2 # b for partially linear model, b2 for interacted model. note that b2 appears in stage1.R for NN
p=length(b(T[1],X[1,]))
p0=ceiling(p/4) 
if (p>60){
  p0=ceiling(p/40)
  
}
D_LB=0 #each diagonal entry of \hat{D} lower bounded by D_LB
D_add=.2 #each diagonal entry of \hat{D} increased by D_add. 0.1 for 0, 0,.2 otw
max_iter=10 #max number iterations in Dantzig selector iteration over estimation and weights

alpha_estimator=1
gamma_estimator=1
lambda_minimax=0
stage1_estimators<-get_stage1(Y.1,D.1,X.1,p0,D_LB,D_add,max_iter,dict,lambda_minimax,alpha_estimator,gamma_estimator)
alpha_hat=stage1_estimators[[1]]

# train - oracle mse
alpha_hat.1=alpha_hat(D.1,X.1)
oracle_mse.1=get_oracle_mse(D.1,pi.1,alpha_true.1,alpha_hat.1)
#oracle_mse.1[[1]]
oracle_mse.1[[2]]

# test - oracle mse
alpha_hat.2=alpha_hat(D.2,X.2)
oracle_mse.2=get_oracle_mse(D.2,pi.2,alpha_true.2,alpha_hat.2)
#oracle_mse.2[[1]]
oracle_mse.2[[2]]

### Dantzig
alpha_estimator=0
gamma_estimator=0
lambda_minimax=0
stage1_estimators<-get_stage1(Y.1,D.1,X.1,p0,D_LB,D_add,max_iter,dict,lambda_minimax,alpha_estimator,gamma_estimator)
alpha_hat=stage1_estimators[[1]]

# train - oracle mse
alpha_hat.1=alpha_hat(D.1,X.1)
oracle_mse.1=get_oracle_mse(D.1,pi.1,alpha_true.1,alpha_hat.1)
#oracle_mse.1[[1]]
oracle_mse.1[[2]]

# test - oracle mse
alpha_hat.2=alpha_hat(D.2,X.2)
oracle_mse.2=get_oracle_mse(D.2,pi.2,alpha_true.2,alpha_hat.2)
#oracle_mse.2[[1]]
oracle_mse.2[[2]]


compare=read.csv("mse_experiment.csv")
# conclusion: oracle mse of minimax riesz is higher

##############################
# mu empirical, lambda=5/n^0.4
##############################

#######################################
# 3 folds: train a, train g, evaluate a
#######################################