########
# set up
########

rm(list=ls())

library("foreign")
library("dplyr")
library("ggplot2")
library("MASS")
library("nnet")	#quicker multinomial logit

setwd("~/Documents/research/minimax_riesz")

#######################
# clean and format data
#######################

#load data
df  <- read.dta("sipp1991.dta")
Y <- df[,"net_tfa"]
D <- df[,"e401"]
X <- cbind(df[,"age"], df[,"inc"], df[,"educ"], df[,"fsize"], df[,"marr"], df[,"twoearn"], df[,"db"], df[,"pira"], df[,"hown"])

summary(X)

#settings
trim=0 # 1 means drop untreated observations with propensities outside of the range of treated propensities
quintile=0 # 0 means keep all observations. 1-5 denotes which quintile of the income distribution

for(trim in 0:1){
  for(quintile in 0:5){
    
    #impose common support
    if(trim){
      X_scaled <- scale(X,center=TRUE,scale=TRUE)
      p.1 <- multinom(D~X_scaled-1, trace=FALSE)$fitted.values
      indexes.to.drop <- which(p.1 < min(p.1[D==1]) | max(p.1[D==1]) < p.1)
      if (length(indexes.to.drop)==0) {indexes.to.drop <- n+1}	#R throws a wobbly if [-indexes.to.drop] is negating an empty set. 
      n.per.treatment <- as.vector(table(D[-indexes.to.drop]))
      n.trim <- n.per.treatment[1]+n.per.treatment[2]
      
      Y.trimmed=Y[-indexes.to.drop]
      D.trimmed=D[-indexes.to.drop]
      X.trimmed=X[-indexes.to.drop,]
    }else{
      Y.trimmed=Y
      D.trimmed=D
      X.trimmed=X
    }
    
    # subset to relevant income quintile
    inc=X.trimmed[,2]
    if (quintile>0){
      q <- ntile(inc, 5)
      Y.q=Y.trimmed[q==quintile]
      D.q=D.trimmed[q==quintile]
      X.q=X.trimmed[q==quintile,]
    } else {
      Y.q=Y.trimmed
      D.q=D.trimmed
      X.q=X.trimmed
    }
    
    out=cbind(Y.q,D.q,X.q)
    out=as.data.frame(out)
    names(out)=c("Y","D","X1","X2","X3","X4","X5","X6","X7","X8","X9")
    
    if(trim){
      write.csv(out,paste0("to_share_401k/quintile",quintile,"_trimmed.csv"))
    }else{
      write.csv(out,paste0("to_share_401k/quintile",quintile,".csv"))
    }
    
  }
}



