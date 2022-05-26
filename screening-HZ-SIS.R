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
#data generation
n=400
p=1000
s=10
true_pos=seq(50,50*s,50)  ## non zero position
set.seed(1)
beta1 = rnorm(s,4,.1)  ### beta true value
sigma=matrix(rep(0,p*p),nrow=p)
for(i in 1:p){for(j in 1:p){sigma[i,j]=.9^abs(i-j)}}
library(MASS)
set.seed(1)
X = mvrnorm(n, rep(0,p),sigma)         ### X generated
relu <- function(x) sapply(x, function(z) max(0,z))
#signal =  2*X[,true_pos[1]] + 2*(X[,true_pos[2]])^3 + exp(X[,true_pos[3]])+6*sin(X[,true_pos[4]])+2*relu((X[,true_pos[5]])^3)  ### signal generated
#signal =  2*X[,true_pos[1]]+ (X[,true_pos[2]])^3 + exp(X[,true_pos[3]])+6*X[,true_pos[4]]*X[,true_pos[5]]  ### signal generated
#signal= relu(X[,true_pos]%*%beta1)
signal= (X[,true_pos]%*%beta1/10)^3+ 3*(X[,true_pos]%*%beta1/10)
vS=var(signal)
vN=1  
set.seed(1) 
error=rnorm(n=n,sd=sqrt(vN))
y=signal+error
unscaled_X=X
X=scale(X)

#_____________________________________________________________________________________________
#nonparanomal and HZ-SIS
all= cbind(y,X)
library(huge)
all=huge.npn(all)

b=(1.25*n)^(1/6)/sqrt(2)
weight_cal=function(k)
{
  d= matrix(rep(0,n*n), ncol=n)
  for(i in 1:n){d[i,]= sapply(1:n, function(j){return((all[i,(k+1)]-all[j,(k+1)])^2 + (all[i,1]-all[j,1])^2)})}
  d1=sapply(1:n, function(i){return((all[i,(k+1)])^2+(all[i,1])^2)})
  w= sum(exp(-b*b*d/2))/(n^2) - 2*sum(exp(-b*b*d1/(2*(1+b*b))))/(n*(1+b*b)) + 1/(1+2*b*b)
  return(w)
}
#weights= sapply(1:p, weight_cal)
library(doParallel)
registerDoParallel(50)
weights=foreach(k=1:p, .combine = c)%dopar%{ weight_cal(k)}
stopImplicitCluster()
active_index=tail(order(weights),7*n/log(n))
intersect(true_pos, active_index)

#Nodewise-lasso
nodewise=function(i)
{
  library(glmnet)
  lasso_fit= glmnet(all[,-i],all[,i],lambda=sqrt(log(p)/n), alpha=1, intercept = F, 
                    standardize = T,penalty.entropyctor = rep(1,(p)))
  lambda_sel=sqrt(log(p)/n)
  tau= sum((all[,i]-all[,-i]%*%coef(lasso_fit, lambda_sel)[-1])^2)/n + 
    lambda_sel*sum(abs(coef(lasso_fit, lambda_sel)[-1]))
  coeff= rep(1,(p+1))
  coeff[-i]=-coef(lasso_fit, lambda_sel)[-1]
  return(c(tau,coeff))
}

nodewise_all=t(sapply(1:(p+1), nodewise))
C= nodewise_all[,-1]
T= diag(1/nodewise_all[,1])
sigma_inv_hat= T%*%C
sigma_inv_hat_x=sigma_inv_hat[-1,-1]



#sigma_inv_hat_lam= huge(cbind(y,all), method="glasso")
#sigma_inv_hat = huge.select(sigma_inv_hat_lam,criterion = "ebic")$opt.icov
#sigma_inv_hat_x= sigma_inv_hat[-1,-1]

#clustering
g=list(NULL)
active=active_index
i=1
while(length(active>=1))
{
    if(is.element(active_index[i],unlist(g))==FALSE){
	group=  which(sigma_inv_hat_x[active_index[i],]!=0)
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
    pos=which(marged_index==0)[which(cor>.8)]
    g[[marged_in]]=sort(unique(unlist(g[c(pos)])))
    if(length(pos>2)){marged_index[pos[which(pos!=marged_in)]]=rep(1, length(pos[which(pos!=marged_in)]))}
  }
  i=i+1
}
#___________________________________________________________________________
g_short=g[marged_index==0]
g1=lapply(1:length(g_short), function(x){return(sort(unique(unlist(lapply(g_short[[x]], function(l){return(which(sigma_inv_hat_x[l,]!=0))})))))})

final_group=g1
p1=length(final_group)
grp_length= sapply(1:length(final_group), function(x){return(length(final_group[[x]]))})
g_unlist=unlist(final_group)
#active_set= sapply(1:length(final_group), function(x){return(final_group[[x]][which.max(abs(sigma_inv_hat[1,-1][final_group[[x]]]))])})
active_set= sapply(1:length(final_group), function(x){return(final_group[[x]][which.max(abs(weights[final_group[[x]]]))])})
return(list(true_pos,grp_length, g_unlist, active_set,  cbind(y,X[,active_set]), cbind(y,unscaled_X)))
}
#output=matrix(rep(0,30),ncol=3)
#for(row in 1:10){output[row,]=mc_iter(row)}
#___________________________________________________________________________________________________________________
ans = mc_iter(1)
setwd("...")
write.csv(ans[1],"supp_true.csv")
write.csv(ans[2],"grp_length.csv")
write.csv(ans[3],"g_unlist.csv")
write.csv(ans[4],"active_set.csv")
write.csv(ans[5],"active_data.csv")
write.csv(ans[6],"all.csv")
#cat(capture.output(print(ans[2]), file="g_short.csv"))
#write.csv(ans,"Output.csv")

quit(save="no")

