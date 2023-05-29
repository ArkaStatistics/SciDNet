rm(list = ls())

## Reads arguments from the command line
if(TRUE){
  args=(commandArgs(TRUE))
  for(i in 1:length(args)){
    eval(parse(text=args[[i]]))
  }
}else{
  jobID=1
}


mc_iter=function(iter)
{
n=600
p=5000
s=5
true_pos=seq(50,450,100)  ## non zero position
#set.seed(1)
beta1 = rnorm(s,1,.0001)  ### beta true value
sigma=matrix(rep(0,p*p),nrow=p)
for(i in 1:p){for(j in 1:p){sigma[i,j]=.9^abs(i-j)}}
library(MASS)
library(mvtnorm)
#set.seed(1)
#X=rmvt(n, sigma , df = 5, delta = rep(0, nrow(sigma)))
X = mvrnorm(n, rep(0,p),sigma)         ### X generated
relu <- function(x) sapply(x, function(z) max(0,z))
#signal =  2*X[,true_pos[1]] + 2*(X[,true_pos[2]])^3 + exp(X[,true_pos[3]])+6*sin(X[,true_pos[4]])+2*relu((X[,true_pos[5]])^3)  ### signal generated
#signal =  2*X[,true_pos[1]]+ (X[,true_pos[2]])^3 + exp(X[,true_pos[3]])+6*X[,true_pos[4]]*X[,true_pos[5]]  ### signal generated
#signal= relu(X[,true_pos]%*%beta1)
signal= (X[,true_pos]%*%beta1/10)^3+ 3*(X[,true_pos]%*%beta1/10)
#signal= X[,true_pos]%*%beta1
vS=var(signal)
vN=10
#set.seed(1) 
error=rnorm(n=n,sd=sqrt(vN))
y=signal+error
unscaled_X=X
X=scale(X)

#_____________________________________________________________________________________________
#fitting lasso 
print("screening starts")
all= cbind(y,X)
library(huge)
all=huge.npn(all)
y_np=all[,1]
Z=all[,-1]
HZ<-c(1:p)
for (k in 1:p){
   print(k)
   beta<-1/sqrt(2)*(n*(2*2+1)/4)^(1/(2+4))
   d<-cbind(y_np,Z[,k])
   diffy<-matrix(rep(y_np,n),n,n,byrow=TRUE)-matrix(rep(y_np,n),n,n,byrow=FALSE)
   diffx<-matrix(rep(Z[,k],n),n,n,byrow=TRUE)-matrix(rep(Z[,k],n),n,n,byrow=FALSE)
   Dij<-diffy^2+diffx^2
   Di<-diag((d)%*%t(d))
   HZ[k]<- sum(exp(-beta*beta/2*Dij))/n-2*(1+beta^2)^(-2/2)*sum(exp(-beta^2/(2*(1+beta^2))*Di))+n*(1+2*beta^2)^(-2/2)
  
}

active_index=tail(order(HZ),4*n/log(n))
intersect(true_pos, active_index)
print("active index selected")
nodewise=function(i)
{
  library(glmnet)
  lasso_fit= glmnet(all[,-i],all[,i],lambda=sqrt(log(p)/n), alpha=1, intercept = F, 
                    standardize = F,penalty.entropyctor = rep(1,(p)))
  lambda_sel=sqrt(log(p)/n)
  tau= sum((all[,i]-all[,-i]%*%coef(lasso_fit, lambda_sel)[-1])^2)/n + 
    lambda_sel*sum(abs(coef(lasso_fit, lambda_sel)[-1]))
  coeff= rep(1,(p+1))
  coeff[-i]=-coef(lasso_fit, lambda_sel)[-1]
  print(i)
  return(c(tau,coeff))
}

nodewise_all=t(sapply(1:(p+1), nodewise))
C= nodewise_all[,-1]
T= diag(1/nodewise_all[,1])
sigma_inv_hat= T%*%C
sigma_inv_hat_x=sigma_inv_hat[-1,-1]


print("sigma_inv estimated")

g=list(NULL)
active=active_index
i=1
while(length(active>=1))
{
    if(is.element(active_index[i],unlist(g))==FALSE){
	group=  which(abs(sigma_inv_hat_x[active_index[i],])>0.05)
    match= which(is.element(active,group)==TRUE)
    g[[i]]=  sort(active[match])
	active=active[-match]
    }
    i=i+ 1
}
all_active= unique(unlist(g))
if(length(which(sapply(1:length(g),function(i){return(length(g[[i]]))})==0))>0){g= g[-which(sapply(1:length(g),function(i){return(length(g[[i]]))})==0)]}

i=1
k=1
marged_index=rep(0, length(g))
while(i<=length(g))
{
  if(marged_index[i]==0){
  cor= sapply(which(marged_index==0), function(j){return(abs(max(cor(X[,g[[i]]],X[,g[[j]]]))))})
  pos=which(marged_index==0)[which(cor>.8)]
  g[[i]]=sort(unique(unlist(g[c(pos)])))
  marged_index[pos[which(pos!=i)]]=rep(1, length(pos[which(pos!=i)]))}  
  if(marged_index[i]==1)
  {
    intersection=rep(0,length(g))
    for(k in 1:length(g))
    {
      if(marged_index[k]==0){intersection[k]= length(intersect(g[[i]], g[[k]]))}
    }
    marged_in=which(intersection>0)
    #marged_in=which( sapply(which(marged_index==0),function(j){return(length(intersect(g[[i]],g[[j]])))})>0)
    cor= sapply(which(marged_index==0), function(j){return(max(abs(cor(X[,g[[marged_in]]],X[,g[[j]]]))))})
    pos=which(marged_index==0)[which(cor>.9)]
    g[[marged_in]]=sort(unique(unlist(g[c(pos)])))
    if(length(pos>2)){marged_index[pos[which(pos!=marged_in)]]=rep(1, length(pos[which(pos!=marged_in)]))}
  }
  print(i)
  i=i+1
}
#___________________________________________________________________________
#cleaning stage
g_short=g[marged_index==0]
g1=lapply(1:length(g_short), function(x){return(sort(unique(unlist(lapply(g_short[[x]], function(l){return(which(sigma_inv_hat_x[l,]!=0))})))))})

print(g_short)
final_group=g_short
p1=length(final_group)
grp_length= sapply(1:length(final_group), function(x){return(length(final_group[[x]]))})
g_unlist=unlist(final_group)
#active_set= sapply(1:length(final_group), function(x){return(final_group[[x]][which.max(abs(sigma_inv_hat[1,-1][final_group[[x]]]))])})
active_set= sapply(1:length(final_group), function(x){return(final_group[[x]][which.max(abs(HZ[final_group[[x]]]))])})
power_screening= length(intersect(true_pos, active_index))/length(true_pos)
FDR_screening= (length(active_index)-length(intersect(true_pos, active_index)))/length(active_index)

library(knockoff)
  knockoffs1 = function(X) create.second_order(X, method='sdp',shrink=F)
  knf_index= knockoff.filter(X[,active_set],y, knockoffs=knockoffs1, statistic=stat.lasso_coefdiff,fdr=.2)$selected
  power_knf= length(intersect(true_pos, c(unlist(final_group[knf_index]))))/length(true_pos)
  FDR_knf= (length(knf_index)-sum(is.element(true_pos,unlist(final_group[knf_index]))))/max(1,length(knf_index))
  size_knf= length(unlist(final_group[knf_index]))
  rbind(c(power_screening, FDR_screening), c(power_knf,FDR_knf))
#return(list(true_pos,grp_length, g_unlist, active_set,  cbind(y,X[,active_set]), cbind(y,unscaled_X), c(power_screening, FDR_screening), c(power_knf,FDR_knf)))
return(list(true_pos,grp_length, g_unlist, active_set,  cbind(y,X[,active_set]), cbind(y,unscaled_X), c(power_screening, FDR_screening), c(power_knf,FDR_knf), c(length(active_index),length(active_set), size_knf, length(knf_index))))

}
#output=matrix(rep(0,30),ncol=3)
#for(row in 1:10){output[row,]=mc_iter(row)}
#___________________________________________________________________________________________________________________
ans = mc_iter(1)
setwd("/mnt/ufs18/home-033/gangulia/Work")
dir_1 = paste0("Rep_",jobID) 
dir.create(dir_1)

setwd(dir_1)
write.csv(ans[1],"supp_true.csv")
write.csv(ans[2],"grp_length.csv")
write.csv(ans[3],"g_unlist.csv")
write.csv(ans[4],"active_set.csv")
write.csv(ans[5],"active_data.csv")
write.csv(ans[6],"all.csv")
write.csv(ans[7],"screening_stat.csv")
write.csv(ans[8],"knf_stat.csv")
write.csv(ans[9],"sizes.csv")

#cat(capture.output(print(ans[2]), file="g_short.csv"))
#write.csv(ans,"Output.csv")

quit(save="no")

