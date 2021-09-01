rm(list=ls())

setwd("~/Documents/research/minimax_riesz")

library("kernlab")
library("MASS")
source('kernel_fast.R')
source('sim.R')
source('minimax_aux.R')
library("gtools")

############
# simulation
############

n=1000
p=10

set.seed(2)

data=get_sim(n,p)

Y=data[[1]]
D=data[[2]]
X=data[[3]]
pi=data[[4]]
alpha_true=data[[5]]

hist(pi)

#######
# train
#######

lambda=10^(-10) #should be like ln(n)/n; validate out of sample
# K1=ss[[1]]
# K2=ss[[2]]
# K3=ss[[3]]
# K4=ss[[4]]
# K=ss[[5]]
# Delta_init=ss[[6]]
# v=ss[[7]]
# V=ss[[8]]

# estimator
ss=get_ss(D,X)
Delta=get_Delta(ss,lambda)
Omega=get_Omega(ss,Delta,lambda)
K_WW=ss[[1]] # eval
alpha_hat=get_alpha(ss,Delta,Omega,K_WW,lambda)

# oracle mse
oracle_mse=get_oracle_mse(D,pi,alpha_true,alpha_hat)
oracle_mse[[1]]
oracle_mse[[2]]

# oracle has access to alpha0. we do not, so we use empirical loss
# alpha_hat = a(w_1),...,a(w_n)
# also need f_hat = f(w_1),...,f(w_n)
# and m_hat = m(w_1;f)...m(w_n;f) = f(1,x_1)-f(0,x_1),...,f(1,x_n)-f(0,x_n)

# empirical mse
g_hat=get_g(ss,Delta,alpha_hat)
K1=ss[[1]]
K2=ss[[2]]
K3=ss[[3]]
K4=ss[[4]]
empirical_mse=get_empirical_mse(K1,K2,K3,K4,alpha_hat,g_hat)
empirical_mse

#######
# tune
#######

# split into two folds, train on one, validate on other
# in particular: a,f trained on fold 1, evaluated on fold 2 to create alpha_hat, f_hat, m_hat

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

# precompute SS
ss.1=get_ss(D.1,X.1)
ss.2=get_ss(D.2,X.2)
ss_eval=get_ss_eval(D.1,D.2,X.1,X.2)

# tuning values
lambda_vals=rep(0,20)
mu_vals=rep(0,20)

for (j in 1:length(lambda_vals)){
  lambda_vals[j]=10^(-j)
  mu_vals[j]=10^(-j)
}
error=matrix(0,length(lambda_vals)*length(mu_vals),10)

for (i in 1:length(lambda_vals)){
  
  lambda=lambda_vals[i]
  paste0("lambda=",lambda)
  
  for(j in 1:length(mu_vals)){
    
    mu=mu_vals[j]
    paste0("mu=",mu)
    idx=(i-1)*length(mu_vals)+j
    error[idx,1]=lambda
    error[idx,2]=mu
    
    tryCatch({
      
      # train - estimator
      Delta.1=get_Delta(ss.1,lambda)
      Omega.1=get_Omega(ss.1,Delta.1,lambda)
      K_WW.1=ss.1[[1]] # eval
      alpha_hat.1=get_alpha(ss.1,Delta.1,Omega.1,K_WW.1,lambda,mu)
      
      # train - oracle mse
      oracle_mse.1=get_oracle_mse(D.1,pi.1,alpha_true.1,alpha_hat.1)
      #oracle_mse.1[[1]]
      error[idx,3]=oracle_mse.1[[2]]
      
      # train - empirical mse
      g_hat.1=get_g(ss.1,Delta.1,alpha_hat.1)
      K1.1=ss.1[[1]]
      K2.1=ss.1[[2]]
      K3.1=ss.1[[3]]
      K4.1=ss.1[[4]]
      empirical_mse.1=get_empirical_mse(K1.1,K2.1,K3.1,K4.1,alpha_hat.1,g_hat.1)
      error[idx,4]=empirical_mse.1
      empirical_mse_abs.1=get_empirical_mse_abs(K1.1,K2.1,K3.1,K4.1,alpha_hat.1,g_hat.1)
      error[idx,5]=empirical_mse_abs.1
      
      # test - estimator
      K_WW.2=ss_eval[[1]] # train on 1, eval on 2
      alpha_hat.2=get_alpha(ss.1,Delta.1,Omega.1,K_WW.2,lambda,mu) # train on 1, eval on 2
      
      # test - oracle mse
      oracle_mse.2=get_oracle_mse(D.2,pi.2,alpha_true.2,alpha_hat.2)
      #oracle_mse.2[[1]]
      error[idx,6]=oracle_mse.2[[2]]
      
      # empirical mse - v1
      K1.2=ss_eval[[1]]
      K2.2=ss_eval[[2]]
      K3.2=ss_eval[[3]]
      K4.2=ss_eval[[4]]
      empirical_mse.2=get_empirical_mse(K1.2,K2.2,K3.2,K4.2,alpha_hat.2,g_hat.1)
      error[idx,7]=empirical_mse.2
      empirical_mse_abs.2=get_empirical_mse_abs(K1.2,K2.2,K3.2,K4.2,alpha_hat.2,g_hat.1)
      error[idx,8]=empirical_mse_abs.2
      
      # test - empirical mse - v2
      Delta.2=get_Delta(ss.2,lambda)
      g_hat.2=get_g(ss.2,Delta.2,alpha_hat.2) # train and eval on 2 given alpha_hat (train on 1, eval on 2)
      K1.2=ss.2[[1]]
      K2.2=ss.2[[2]]
      K3.2=ss.2[[3]]
      K4.2=ss.2[[4]]
      empirical_mse.2=get_empirical_mse(K1.2,K2.2,K3.2,K4.2,alpha_hat.2,g_hat.2)
      error[idx,9]=empirical_mse.2
      empirical_mse_abs.2=get_empirical_mse_abs(K1.2,K2.2,K3.2,K4.2,alpha_hat.2,g_hat.2)
      error[idx,10]=empirical_mse_abs.2
      
      
    }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
  }
    
}

error=as.data.frame(error)
names(error)<-c("lambda","mu",
                "train_oracle_mse",
                "train_empirical_mse","train_empirical_mse_abs",
                "test_oracle_mse",
                "test_empirical_mse_v1","test_empirical_mse_v1_abs",
                "test_empirical_mse_v2","test_empirical_mse_v2_abs")
error

write.csv(error,"mse_experiment.csv")

nonzero=error[error$test_oracle_mse>0,]
ideal=which.min(nonzero$test_oracle_mse)
nonzero[ideal,]

ideal=which.min(nonzero$test_oracle_mse)
nonzero[ideal,]

# is adversary over train or test?
# argument for train (i): 
# ---this is the function in anticipation of which alpha_hat is a BR
# ---regularization is only for training. don't know closed form for test set since no ridge
# argument for test (ii): 
# ---this is the adversary given alpha_hat on the evaluation samples
# ---more reasonable values

# so far, oracle test suggest lambda=10^{-10}
# so far, empirical mse is mononically increasing in lambda...

######
# eval
######

lambda_star=10^(-10)
d=as.matrix(1)
x=t(colMeans(X))

# estimator
ss=get_ss(D,X)
Delta=get_Delta(ss,lambda_star)
Omega=get_Omega(ss,Delta,lambda_star)

K_Dd=get_K_eval(D,d)
K_Xx=get_K_eval(X,x)
K_Ww=K_Dd*K_Xx
K_wW=t(K_Ww) # eval

alpha_hat=get_alpha(ss,Delta,Omega,K_wW,lambda_star)