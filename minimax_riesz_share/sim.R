get_sim <- function(n,p){
  
  #nu=rnorm(n)
  eps=rnorm(n)
  
  beta=rep(0,p)
  for (j in 1:p){
    beta[j]=1/j^2
  }
  
  Sigma=diag(p)
  for (i in 1:p){
    for (j in 1:p){
      if (abs(i-j)==1)
        Sigma[i,j]=0.5
    }
  }
  Mu=rep(0,p) 
  X=mvrnorm(n,Mu,Sigma)
  
  #D=(3*(X%*%beta)-0.75*nu > 0)+0 - v1 extreme propensities
  #D=((X%*%beta)-nu > 0)+0 - v2 reasonable propensities
  pi=inv.logit(X%*%beta,min=0.05,max=0.95) # v3 propensities away from zero and one
  
  D=rbinom(n,1,pi)
  Y=1.2*(D+X%*%beta)+D^2+D*X[,1]+eps
  
  #pi=pnorm(4*X %*% beta) # since E[D|X]=Pr(3X'beta-3/4 nu>0|X)=Pr(4X'beta-nu>0|X)=Pr(nu<4X'beta|X)
  #pi=pnorm(X %*% beta) # since E[D|X]=Pr(X'beta-nu>0|X)=Pr(nu<X'beta|X)
  #hist(pi,breaks=50)
  alpha_true=D/pi-(1-D)/(1-pi)
  gamma_true=1.2*(D+X%*%beta)+D^2+D*X[,1]
  
  D=as.matrix(D)
  
  data=list(Y,D,X,pi,alpha_true,gamma_true)
  
  return(data) #pass around data as list
}

mse<-function(v1,v2){
  n=length(v1)
  out=crossprod(v1-v2)/n
  return(out)
}
