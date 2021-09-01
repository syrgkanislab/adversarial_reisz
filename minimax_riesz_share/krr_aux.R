get_ss_krr<-function(D,X){
  
  n=dim(X)[1]
  
  # kernel matrices
  K_DD=get_K(D)
  K_XX=get_K(X)
  
  K_D1=get_K_eval(D,as.matrix(1))
  K_D0=get_K_eval(D,as.matrix(0))
  
  K1=K_DD*K_XX
  K2=rep.col(K_D1-K_D0,n)*K_XX
  K3=rep.row(K_D1-K_D0,n)*K_XX
  K4=(k_bin(1,1)-k_bin(1,0)-k_bin(0,1)+k_bin(0,0))*K_XX
  
  #hist(K2-t(K3)) # sanity check
  
  # sufficient stats
  K=rbind(cbind(K1,K2),cbind(K3,K4))
  Omega=rbind(cbind(K1%*%K1,K1%*% K2),cbind(K3%*%K1,K3%*%K2))
  v=n*as.matrix(c(rowMeans(K2),rowMeans(K4))) #Psi \hat{mu}^m
  u=rbind(K1,K3)
  
  return(list(K_DD,K_XX,K,Omega,v))
}

get_ss_krr_eval<-function(D.1,D.2,X.1,X.2){
  
  n.1=dim(X.1)[1]
  n.2=dim(X.2)[1]
  
  # kernel matrices
  K_DD=get_K_eval(D.1,D.2)
  K_XX=get_K_eval(X.1,X.2)
  
  K_D1.1=get_K_eval(D.1,as.matrix(1))
  K_D0.1=get_K_eval(D.1,as.matrix(0))
  
  K_D1.2=get_K_eval(D.2,as.matrix(1))
  K_D0.2=get_K_eval(D.2,as.matrix(0))
  
  K1=K_DD*K_XX
  K2=rep.col(K_D1.2-K_D0.2,n.1)*K_XX # Phi^2 Psi^1'
  K3=rep.row(K_D1.1-K_D0.1,n.2)*K_XX # Phi^{m,2} Psi^1'
  K4=(k_bin(1,1)-k_bin(1,0)-k_bin(0,1)+k_bin(0,0))*K_XX
  
  return(list(K1,K2))
  
}