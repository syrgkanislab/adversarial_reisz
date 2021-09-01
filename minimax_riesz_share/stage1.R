#stage 1 function
l=0.1

RMD_dantzig <- function(M, G, D, lambda=0, sparse = TRUE) {
  
  p <- ncol(G)
  zp <- rep(0, p)
  L <-c(l,rep(1,p-1)) #dictionary is ordered (constant,T,X,TX)
  
  A <- solve(diag(D),G)
  R <- rbind(A, -A)
  
  a <- solve(diag(D),M)
  r <- c(a - lambda*L, -a - lambda*L)
  
  if(sparse) {
    Ip <- as(p, "matrix.diag.csr")
    R <- as.matrix.csr(R)
    f <- rq.fit.sfnc(Ip, zp, R = R, r = r)
  } else {
    Ip <- diag(p)
    f <- rq.fit.fnc(Ip, zp, R = R, r = r)
  }
  
  return(f)
}

RMD_lasso <- function(M, G, D, lambda=0, control = list(maxIter = 1000, optTol = 10^(-5), 
                                                        zeroThreshold = 10^(-6)), beta.start = NULL) {
  
  p <- ncol(G)
  
  Gt<-G
  Mt<-M
  
  L <-c(l,rep(1,p-1)) #dictionary is ordered (constant,...)
  lambda_vec=lambda*L*D #v3: insert D here
  
  if (is.null(beta.start)) {
    beta <- rep(0,p) #vs low-dimensional initialization
  }
  else {
    beta <- beta.start
  }
  wp <- beta
  mm <- 1
  while (mm < control$maxIter) {
    beta_old <- beta
    for (j in 1:p) {
      rho=Mt[j]-Gt[j,]%*%beta+Gt[j,j]*beta[j]
      z=Gt[j,j]
      
      if (sum(is.na(rho)) >= 1) {
        beta[j] <- 0
        next
      }
      if (rho < -1 * lambda_vec[j]) 
        beta[j] <- (rho+lambda_vec[j])/z
      if (abs(rho) <= lambda_vec[j]) 
        beta[j] <- 0
      if (rho > lambda_vec[j]) 
        beta[j] <- (rho-lambda_vec[j])/z
    }
    wp <- cbind(wp, beta)
    if (sum(abs(beta - beta_old), na.rm = TRUE) < control$optTol) {
      break
    }
    mm <- mm + 1
  }
  w <- beta
  w[abs(w) < control$zeroThreshold] <- 0
  return(list(coefficients = w, coef.list = wp, num.it = mm))
}


get_D <- function(Y,T,X,m,rho_hat,b){
  n=nrow(X)
  p=length(b(T[1],X[1,]))
  
  df=matrix(0,p,n)
  for (i in 1:n){
    df[,i]=b(T[i],X[i,])*as.vector(rho_hat %*% t(b(T[i],X[i,])))-m(Y[i],T[i],X[i,],b)
  }
  df=df^2
  D2=rowMeans(df)
  
  D=sqrt(D2)
  return(D) #pass around D as vector
}

c=0.5
alpha=0.1
tol=1e-6

RMD_stable<-function(Y,T,X,p0,D_LB,D_add,max_iter,b,is_alpha,is_lasso){
  
  k=1
  
  p=length(b(T[1],X[1,]))
  n=length(T)
  
  # low-dimensional moments
  X0=X[,1:p0]
  MNG0<-get_MNG(Y,T,X0,b)
  M_hat0=MNG0[[1]]
  N_hat0=MNG0[[2]]
  G_hat0=MNG0[[3]]
  
  # initial estimate
  rho_hat0=solve(G_hat0,M_hat0)
  rho_hat=c(rho_hat0,rep(0,p-ncol(G_hat0)))
  beta_hat0=solve(G_hat0,N_hat0)
  beta_hat=c(beta_hat0,rep(0,p-ncol(G_hat0)))
  
  # moments
  MNG<-get_MNG(Y,T,X,b)
  M_hat=MNG[[1]]
  N_hat=MNG[[2]]
  G_hat=MNG[[3]]
  
  # penalty
  lambda=c*qnorm(1-alpha/(2*p))/sqrt(n) # snippet
  
  if(is_alpha){ 
    ###########
    # alpha_hat
    ###########
    diff_rho=1
    while(diff_rho>tol & k<=max_iter){
      
      # previous values
      rho_hat_old=rho_hat+0
      
      # normalization
      D_hat_rho=get_D(Y,T,X,m,rho_hat_old,b)
      D_hat_rho=pmax(D_LB,D_hat_rho)
      D_hat_rho=D_hat_rho+D_add
      
      # RMD estimate
      if(is_lasso){
        rho_hat=RMD_lasso(M_hat, G_hat, D_hat_rho, lambda)$coefficients
      }else{
        rho_hat=RMD_dantzig(M_hat, G_hat, D_hat_rho, lambda)$coefficients
      }
      
      # difference
      diff_rho=two.norm(rho_hat-rho_hat_old)
      k=k+1
      
    }
    
    print(paste0('k: '))
    print(paste0(k))
    return(rho_hat)
    
  } else { 
    ###########
    # gamma_hat
    ###########
    diff_beta=1
    while(diff_beta>tol & k<=max_iter){
      
      # previous values
      beta_hat_old=beta_hat+0
      
      # normalization
      D_hat_beta=get_D(Y,T,X,m2,beta_hat_old,b)
      D_hat_beta=pmax(D_LB,D_hat_beta)
      D_hat_beta=D_hat_beta+D_add
      
      # RMD estimate
      if(is_lasso){
        beta_hat=RMD_lasso(N_hat, G_hat, D_hat_beta, lambda)$coefficients
      }else{
        beta_hat=RMD_dantzig(N_hat, G_hat, D_hat_beta, lambda)$coefficients
      }
      
      # difference
      diff_beta=two.norm(beta_hat-beta_hat_old)
      k=k+1
      
    }
    
    print(paste0('k: '))
    print(paste0(k))
    return(beta_hat)
    
  }
}

arg_Forest<- list(clas_nodesize=1, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
arg_Nnet<- list(size=8,  maxit=1000, decay=0.01, MaxNWts=10000,  trace=FALSE)

cv_krr<-function(iota,K,y){
  n=length(y)
  H=diag(n)-K %*% solve(K+iota*diag(n))
  H_tilde=diag(H)
  vec=diag(1/H_tilde) %*% H %*% y
  cv=crossprod(vec)
  return(cv)
}
get_cv_krr<-function(K,y,iota_vals){
  
  n_vals=length(iota_vals)
  cv_vals=rep(0,n_vals)
  
  for (i in 1:n_vals){
    cv_vals[i]=cv_krr(iota_vals[i],K,y)
  }
  
  iota_star=iota_vals[which.min(cv_vals)]
  return(iota_star)
}
iota_vals=10^(seq(-20,-10,1))

#lambda_star=10^(-10)

get_stage1<-function(Y,T,X,p0,D_LB,D_add,max_iter,b,lambda_minimax,mu_minimax,alpha_estimator,gamma_estimator){
  
  p=length(b(T[1],X[1,]))
  n=length(T)
  MNG<-get_MNG(Y,T,X,b)
  B=MNG[[4]]
  
  ###########
  # alpha hat
  ###########
  if(alpha_estimator==0){ # dantzig
    
    rho_hat=RMD_stable(Y,T,X,p0,D_LB,D_add,max_iter,b,1,0)
    alpha_hat<-function(d,z){
      return(b(d,z)%*%rho_hat)
    }
    
  } else if(alpha_estimator==1){ # lasso
    
    rho_hat=RMD_stable(Y,T,X,p0,D_LB,D_add,max_iter,b,1,1)
    alpha_hat<-function(d,z){
      return(b(d,z)%*%as.matrix(rho_hat))
    }
    
  } else if(alpha_estimator==2){ # minimax
    
    D=as.matrix(T)
    X=as.matrix(X)
    
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
    
  } else if(alpha_estimator==3){ # debiased kernel
    
    D=as.matrix(T)
    X=as.matrix(X)
    
    ss=get_ss_krr(D,X)
    
    K_DD=ss[[1]]
    K_XX=ss[[2]]
    K=ss[[3]] #like minimax, not like KRR CEF
    Omega=ss[[4]]
    v=ss[[5]]
    
    iota_star=get_cv_krr(K_DD*K_XX,Y,iota_vals) #LOOCV for KRR CEF
    beta=solve(Omega+iota_star*n*K) %*% v
    
    alpha_hat<-function(d,z){
      
      d=as.matrix(d)
      x=as.matrix(z)
      
      ss_eval=get_ss_krr_eval(D,d,X,x)
      u1=t(ss_eval[[1]])
      u2=t(ss_eval[[2]])
      u=rbind(u1,u2)
      
      Y_hat=t(crossprod(beta,u))
      
      return(Y_hat)
    }
  }
  
  ###########
  # gamma hat
  ###########
  if(gamma_estimator==0){ # dantzig
    
    beta_hat=RMD_stable(Y,T,X,p0,D_LB,D_add,max_iter,b,0,0)
    gamma_hat<-function(d,z){
      return(b(d,z)%*%beta_hat)
    }
    
  } else if(gamma_estimator==1){ # lasso
    
    beta_hat=RMD_stable(Y,T,X,p0,D_LB,D_add,max_iter,b,0,1)
    gamma_hat<-function(d,z){ 
      return(b(d,z)%*%as.matrix(beta_hat))
    }
    
  } else if(gamma_estimator==2){ # random forest
    
    forest<- do.call(randomForest, append(list(x=B,y=Y), arg_Forest))
    gamma_hat<-function(d,z){
      return(predict(forest,newdata=b(d,z), type="response"))
    }
    
  } else if(gamma_estimator==3){ # neural net
    
    # scale down, de-mean, run NN, scale up, remean so that NN works well
    maxs_B <- apply(B, 2, max)
    mins_B <- apply(B, 2, min)
    
    maxs_Y<-max(Y)
    mins_Y<-min(Y)
    
    # hack to ensure that constant covariates do not become NA in the scaling
    const=maxs_B==mins_B
    keep=(1-const)*(1:length(const))
    
    NN_B<-B
    NN_B[,keep]<-scale(NN_B[,keep], center = mins_B[keep], scale = maxs_B[keep] - mins_B[keep])
    
    NN_Y<-scale(Y, center = mins_Y, scale = maxs_Y - mins_Y)
    
    nn<- do.call(nnet, append(list(x=NN_B,y=NN_Y), arg_Nnet))
    gamma_hat<-function(d,z){
      
      NN_b<-b(d,z)
      NN_b[,keep]<-scale(NN_b[,keep], 
                         center = mins_B[keep], 
                         scale = maxs_B[keep] - mins_B[keep])
      
      NN_Y_hat<-predict(nn,newdata=NN_b)
      Y_hat=NN_Y_hat*(maxs_Y-mins_Y)+mins_Y
      
      return(Y_hat)
    }
    
  } else if(gamma_estimator==4){ # KRR
    
    D=as.matrix(T)
    X=as.matrix(X)
    
    K_DD=get_K(D)
    K_XX=get_K(X)
    K=K_DD*K_XX
    iota_star=get_cv_krr(K,Y,iota_vals) #LOOCV
    beta=solve(K+iota_star*diag(n)) %*% Y
    
    gamma_hat<-function(d,z){
      
      d=as.matrix(d)
      x=as.matrix(z)
      
      K_Dd=get_K_eval(D,d)
      K_Xx=get_K_eval(X,x)
      K_Ww=K_Dd*K_Xx
      Y_hat=t(crossprod(beta,K_Ww))
      
      return(Y_hat)
    }
    
  }
  
  return(list(alpha_hat,gamma_hat))
  
}
