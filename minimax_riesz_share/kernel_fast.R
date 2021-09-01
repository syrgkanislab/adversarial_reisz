#########
# kernels
#########

# gaussian kernel
k<-function(v1,v2,lengthscale) {
  out=exp(-1/2*(v1-v2)^2/(lengthscale^2))
  return(out)
}

k_bin<-function(v1,v2) {
  out=as.numeric(v1==v2)
  return(out)
}

###########
# auxiliary
###########

# helpers
rep.row<-function(x,n){
  matrix(rep(x,each=n),nrow=n)
}
rep.col<-function(x,n){
  matrix(rep(x,each=n), ncol=n, byrow=TRUE)
}

# calculate median interpoint distance
get_interpoint<-function(v) {
  n=length(v)
  V_col=rep.col(v,n)
  V_row=rep.row(v,n)
  interpoint=abs(V_col-V_row)
  out=median(interpoint)
  return(out)
}

###################
# kernel operations
###################

# kernel over scalar
get_K_scalar<-function(v) {
  
  if (length(unique(v))>2){ # if non-binary
    lengthscale=get_interpoint(v)
    sigma=1/(2*lengthscale^2)
    kernel=rbfdot(sigma)
  } else {
    kernel=k_bin
  }
  
  K=kernelMatrix(kernel,v)

  return(K)
}

# kernel over vector
get_K<-function(V){
  n=nrow(V)
  p=ncol(V)
  K=matrix(1,n,n) # init as 1 since product kernel
  for (j in 1:p){
    K_new=get_K_scalar(V[,j])
    K=K*K_new # product kernel
  }
  return(K)
}

############
# evaluation
############

# kernel over scalar
get_K_scalar_eval<-function(v,v_eval) {
  
  if (length(unique(v))>2){ # if non-binary
    lengthscale=get_interpoint(v)
    sigma=1/(2*lengthscale^2)
    kernel=rbfdot(sigma)
  } else {
    kernel=k_bin
  }
  
  K=kernelMatrix(kernel,v,v_eval)
  
  return(K)
}

# kernel over vector
get_K_eval<-function(V,V_eval){
  n=nrow(V)
  n_eval=nrow(V_eval)
  p=ncol(V)
  K=matrix(1,n,n_eval) # init as 1 since product kernel
  for (j in 1:p){
    K_new=get_K_scalar_eval(V[,j],V_eval[,j])
    K=K*K_new # product kernel
  }
  return(K)
}