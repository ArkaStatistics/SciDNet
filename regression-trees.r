library(rsample)     # data splitting 
library(dplyr)       # data wrangling
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(caret)       # bagging
setwd("...")
data= read.csv("...csv")[,-1]
y=data[,1]
X=data[,-1]
n=nrow(X)
p=ncol(X)
#for prediction
testing= function(iter)
{
  train_index= sample(1:n, 9*n/10, replace = FALSE)
  data_train= data[train_index,]
  data_test= data[-train_index,]
  ctrl <- trainControl(method = "cv",  number = 10) 
  bagged_cv <- train(
    y,X,
    method = "treebag",
    trControl = ctrl,
    importance = TRUE
  )
  # assess results  
  pred <- predict(bagged_cv, newdata = data_test)
  error= mean((data_test$y-pred)^2)
  cor=cor(data_test$y,pred)


   return(c(error,cor))
}
test_error=matrix(rep(0,200), ncol=2)
for(iter in 1:100){test_error[iter,]= testing(iter)
print(iter, test_error[iter,])}
apply(test_error, 2, mean)
apply(test_error,2,sd)
library(doParallel)
registerDoParallel(15)
test_error=foreach(iter=1:100, .combine = rbind)%dopar%{ 
  testing(iter)
}
stopImplicitCluster()
