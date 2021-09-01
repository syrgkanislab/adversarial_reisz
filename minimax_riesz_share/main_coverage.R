########
# set up
########

rm(list=ls())

#library("foreign")
#library("dplyr")
#library("ggplot2")
library("quantreg")
library("kernlab")

library("MASS")
#library("glmnet")	#lasso, group lasso, and ridge, for outcome models, LPM, mlogit. Also, unpenalized mlogit via glmnet is far faster than the mlogit package.
#library("grplasso")	#glmnet group lasso requires the same n per group (ie multi-task learning), which is perfect for mlogit but wrong for the outcome model.
# library("mlogit")	#slower than unpenalized estimation in glmnet, but glmnet won't fit only an intercept
#library("nnet")	#quicker multinomial logit
#library("randomForest")
#library("gglasso")
#library("plotrix")
#library("gridExtra")
library("gtools")

setwd("~/Documents/research/minimax_riesz")

##################
# helper functions
##################

source('primitives.R')
source('kernel_fast.R')
source('minimax_aux.R')
source('stage1.R')

source('sim.R')

n=1000
p=10
data=get_sim(n,p)
T=data[[2]]
X=data[[3]]

pi=data[[4]]
alpha_true=data[[5]]
summary(alpha_true)

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

source('stage2.R')

############
# simulation
############

set.seed(2)

source('sim.R')

n=1000
p=10

n_sim=100
bias=0

test_CB<-function(theta_hat,se){
  LB=theta_hat-1.96*se
  UB=theta_hat+1.96*se
  out=(LB <= 2.2) & (2.2 <= UB)
  return(out)
}


# tuning values
lambda_vals=rep(NA,7)
mu_vals=rep(NA,7)

for (j in 1:length(lambda_vals)){
  lambda_vals[j]=10^(-2*j)
  mu_vals[j]=10^(-2*j)
}

experiments=matrix(NA,n_sim*length(lambda_vals)*length(mu_vals),6)
colnames(experiments)=c("simulation","lambda","mu","ate","se","coverage")

idx=0

for (i in 1:n_sim){
  
  print(paste0("simulation: ",i))
  
  data=get_sim(n,p)
  Y=data[[1]]
  T=data[[2]]
  X=data[[3]]
  
 for (j in 1:length(lambda_vals)){
   
   lambda=lambda_vals[j]
   print(paste0("lambda=",lambda))
   
   for (k in 1:length(mu_vals)){
     
     idx=idx+1
     
     mu=mu_vals[k]
     print(paste0("mu=",mu))
     
     print(paste0("idx=",idx))
     
     experiments[idx,1]=i
     experiments[idx,2]=lambda
     experiments[idx,3]=mu
     
     alpha_estimator=2
     gamma_estimator=0
     
     tryCatch({
       
       results<-rrr(Y,T,X,p0,D_LB,D_add,max_iter,dict,lambda_minimax=lambda,mu_minimax=mu,alpha_estimator,gamma_estimator,bias)
       
       ate=results[3]
       se=results[4]
       
       experiments[idx,4]=ate
       experiments[idx,5]=se
       experiments[idx,6]=test_CB(ate,se)
       
     }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
     
   }
   
 }
  
  write.csv(experiments,"coverage_experiment.csv")
}

