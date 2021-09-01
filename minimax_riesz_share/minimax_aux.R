get_ss<-function(D,X){
  
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
  Delta_init=rbind(cbind(K1%*%K1,K1%*% K2),cbind(K3%*%K1,K3%*%K2))
  v=rbind(K1%*%K1,K3%*%K1)
  V=as.matrix(c(rowMeans(K2),rowMeans(K4))) #Psi \hat{mu}^m
  
  return(list(K1,K2,K3,K4,K,Delta_init,v,V))
}

#ss=get_ss(D,X)
# K1=ss[[1]]
# K2=ss[[2]]
# K3=ss[[3]]
# K4=ss[[4]]
# K=ss[[5]]
# Delta_init=ss[[6]]
# v=ss[[7]]
# V=ss[[8]]

# estimator
get_Delta<-function(ss,lambda){
  
  K=ss[[5]]
  Delta_init=ss[[6]]
  n=nrow(K)/2
  
  Delta=Delta_init+n*lambda*K
  return(Delta)
}

get_Omega<-function(ss,Delta,lambda){
  
  K=ss[[5]]
  Delta_init=ss[[6]]
  v=ss[[7]]
  n=nrow(K)/2
  
  Omega=t(v)-0.5*t(v) %*% solve(Delta) %*% Delta_init-n*lambda/2*t(v)%*% solve(Delta) %*% K
  return(Omega)
}

get_alpha<-function(ss,Delta,Omega,K_WW,lambda,mu=lambda*6){
  
  K1=ss[[1]]
  v=ss[[7]]
  V=ss[[8]]
  #mu=lambda*6
  n=nrow(K1)
  
  alpha_hat=K_WW %*% solve(1/n*Omega %*% solve(Delta) %*% v + 2*mu*K1) %*% Omega %*% solve(Delta) %*% V
  return(alpha_hat)
}

get_oracle_mse<-function(D,pi,alpha_true,alpha_hat){
  df=as.data.frame(cbind(D,pi,alpha_true,alpha_hat))
  names(df)<-c("treatment","propensity","true_RR","est_RR")
  df<-round(df,3)
  
  trimmed=df[!is.nan(alpha_true),]
  trimmed
  out=mse(trimmed$est_RR,trimmed$true_RR)
  return(list(trimmed, out))
}

get_g<-function(ss,Delta,alpha_hat){
  
  K1=ss[[1]]
  K3=ss[[3]]
  V=ss[[8]]
  n=nrow(K1)
  
  g_hat=0.5 * solve(Delta) %*% (n*V-rbind(K1,K3) %*% alpha_hat)
  return(g_hat)
}

get_empirical_mse<-function(K1,K2,K3,K4,alpha_hat,g_hat){
  f_hat=cbind(K1,K2) %*% g_hat # Phi Psi' gamma_hat
  m_hat=cbind(K3,K4) %*% g_hat # Phi^{m} Psi' gamma_hat
  loss=mean(m_hat-alpha_hat*f_hat-f_hat^2)
  return(loss)
}

get_empirical_mse_abs<-function(K1,K2,K3,K4,alpha_hat,g_hat){
  f_hat=cbind(K1,K2) %*% g_hat # Phi Psi' gamma_hat
  m_hat=cbind(K3,K4) %*% g_hat # Phi^{m} Psi' gamma_hat
  loss=abs(mean(m_hat-alpha_hat*f_hat))
  return(loss)
}

get_ss_eval<-function(D.1,D.2,X.1,X.2){ # check this
  
  n.1=dim(X.1)[1]
  n.2=dim(X.2)[1]
  
  # kernel matrices
  K_DD=get_K_eval(D.2,D.1)
  K_XX=get_K_eval(X.2,X.1)
  
  K_D1.1=get_K_eval(D.1,as.matrix(1))
  K_D0.1=get_K_eval(D.1,as.matrix(0))
  
  K_D1.2=get_K_eval(D.2,as.matrix(1))
  K_D0.2=get_K_eval(D.2,as.matrix(0))
  
  K1=K_DD*K_XX
  K2=rep.col(K_D1.2-K_D0.2,n.1)*K_XX # Phi^2 Psi^1'
  K3=rep.row(K_D1.1-K_D0.1,n.2)*K_XX # Phi^{m,2} Psi^1'
  K4=(k_bin(1,1)-k_bin(1,0)-k_bin(0,1)+k_bin(0,0))*K_XX
  
  return(list(K1,K2,K3,K4))
  
}