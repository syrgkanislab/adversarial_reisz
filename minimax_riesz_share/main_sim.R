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

############
# simulation
############

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

summary(pi)
hist(pi,breaks=50)

##################
# helper functions
##################

source('primitives.R')
source('kernel_fast.R')
source('minimax_aux.R')
source('krr_aux.R')
source('stage1.R')

# dictionary
dict=b2 # b for partially linear model, b2 for interacted model
p=length(b(T[1],X[1,]))

#p0=dim(X0) used in low-dim dictionary in the stage 1 tuning procedure
p0=ceiling(p/4) 
if (p>60){
  p0=ceiling(p/40)
  
}

D_LB=0 #each diagonal entry of \hat{D} lower bounded by D_LB
D_add=.2 #each diagonal entry of \hat{D} increased by D_add. 0.1 for 0, 0,.2 otw
max_iter=10 #max number iterations in Dantzig selector iteration over estimation and weights

###########
# algorithm
###########

#set.seed(1) # for sample splitting

alpha_estimator=3
gamma_estimator=4
bias=0
#alpha_estimator: 0 dantzig, 1 lasso, 2 minimax, 3 krr
#gamma_estimator: 0 dantzig, 1 lasso, 2 rf, 3 nn, 4 krr

lambda=10^(-10)
mu=10^(-10)

source('stage2.R')
results<-rrr(Y,T,X,p0,D_LB,D_add,max_iter,dict,lambda_minimax=lambda,mu_minimax=mu,alpha_estimator,gamma_estimator,bias)
printer(results)
for_tex(results)