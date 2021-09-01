########
# set up
########

rm(list=ls())

library("foreign")
library("dplyr")
library("ggplot2")
library("quantreg")
library("kernlab")

library("MASS")
library("glmnet")	#lasso, group lasso, and ridge, for outcome models, LPM, mlogit. Also, unpenalized mlogit via glmnet is far faster than the mlogit package.
library("grplasso")	#glmnet group lasso requires the same n per group (ie multi-task learning), which is perfect for mlogit but wrong for the outcome model.
# library("mlogit")	#slower than unpenalized estimation in glmnet, but glmnet won't fit only an intercept
library("nnet")	#quicker multinomial logit
library("randomForest")
library("gglasso")
library("plotrix")
library("gridExtra")
library("gtools")

setwd("~/Documents/research/minimax_riesz")

set.seed(2)

source('sim.R')

n=1000
p=10

data=get_sim(n,p)

Y=data[[1]]
T=data[[2]]
X=data[[3]]
pi=data[[4]]
alpha_true=data[[5]]
gamma_true=data[[6]]

hist(pi)

source('kernel_fast.R')
source('minimax_aux.R')

D=as.matrix(T)
X=as.matrix(X)

lambda_minimax=10^(-10)
mu_minimax=10^(-10)

ss=get_ss(D,X)
Delta=get_Delta(ss,lambda_minimax)
Omega=get_Omega(ss,Delta,lambda_minimax)

alpha_hat<-function(d,z){
  
  d=as.matrix(d)
  x=as.matrix(z)
  
  K_Dd=get_K_eval(D,d)
  K_Xx=get_K_eval(X,x)
  K_Ww=K_Dd*K_Xx
  K_wW=t(K_Ww)
  
  Y_hat=get_alpha(ss,Delta,Omega,K_wW,lambda_minimax,mu_minimax)
  
  return(Y_hat)
  
}

Y_hat=alpha_hat(D,X)
dev.new(width=11, height=8, unit="in")
plot(alpha_true,Y_hat)


