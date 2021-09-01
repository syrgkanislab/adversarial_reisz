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

set.seed(1)

n=2000
p=10

n_sim=100

for (i in 1:n_sim){
  
  data=get_sim(n,p)
  
  Y=data[[1]]
  T=data[[2]]
  X=data[[3]]
  pi=data[[4]]
  alpha_true=data[[5]]
  gamma_true=data[[6]]
  
  out=cbind(Y,T,X,pi,alpha_true,gamma_true)
  out=as.data.frame(out)
  names(out)=c("Y","D","X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","propensity","rr","cef")
  
  write.csv(out,paste0("to_share/sim_",i,".csv"))
}


hist(pi,breaks=10)
