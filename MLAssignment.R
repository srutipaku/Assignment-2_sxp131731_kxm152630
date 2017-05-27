# load libraries
library(caret)
# load the dataset
data(iris)
# summarize data
summary(iris[,1:5])

print(NAValues<- is.na(iris))
print(duplicated(iris))
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:5], method=c("scale"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:5])
# summarize the transformed dataset
summary(transformed)


###############################################3
## correlation Plots #############################
panel.cor <- function(x, y, digits=2, prefix="", cex.cor) 
{
  usr <- par("usr"); on.exit(par(usr)) 
  par(usr = c(0, 1, 0, 1)) 
  r <- abs(cor(x, y)) 
  txt <- format(c(r, 0.123456789), digits=digits)[1] 
  txt <- paste(prefix, txt, sep="") 
  if(missing(cex.cor)) cex <- 0.8/strwidth(txt) 
  
  test <- cor.test(x,y) 
  # borrowed from printCoefmat
  Signif <- symnum(test$p.value, corr = FALSE, na = FALSE, 
                   cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                   symbols = c("***", "**", "*", ".", " ")) 
  
  text(0.5, 0.5, txt, cex = cex * r) 
  text(.8, .8, Signif, cex=cex, col=2) 
}

pairs(transformed[1:5], lower.panel=panel.smooth, upper.panel=panel.cor)
transformed$Species<-as.numeric(transformed$Species)
hist(cor(transformed))
########################################
####Neural Nets#########################
#########################################
library(neuralnet)
b <- transformed
b$Species <- as.numeric(b$Species)
apply(b, MARGIN = 2, FUN = function(x) sum(is.na(x)))
maxs = apply(b, MARGIN = 2, max)
mins = apply(b, MARGIN = 2, min)
scaled = as.data.frame(scale(b, center = mins, scale = maxs - mins))
trainIndex <- sample(1:nrow(scaled), 0.8 * nrow(scaled))
train <- scaled[trainIndex, ]
test <- scaled[-trainIndex, ]

###Train the data

n <- names(train)
f <- as.formula(paste("Species ~", paste(n[!n %in% "Species"], collapse = " + ")))
#nn <- neuralnet(f,data=train,hidden=c(2,7),threshold = 0.05,rep=5)


######################################################
########## For Perceptron#############################
# for perceptron change the hidden value in neural net to 0
#nn <- neuralnet(f,data=train,hidden=0,threshold = 0.05,rep=5)
#

plot(nn)
nn$result.matrix

pred <- compute(nn,test[,1:4])
pred.scaled <- pred$net.result *(max(b$Species)-min(b$Species))+min(b$Species)
real.values <- (test$Species)*(max(b$Species)-min(b$Species))+min(b$Species)
MSE.nn <- sum((real.values - pred.scaled)^2)/nrow(test)
print(MSE.nn)
plot(real.values, pred.scaled, col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')


#####################################################################
## Decision TREEE ###################################################
#####################################################################
library(rpart)
library(rpart.plot)
n <- names(train)
f <- as.formula(paste("Species ~", paste(n[!n %in% "Species"], collapse = " + ")))

fit <- rpart(f, data = train, method =
               'class', parms = list(split = "information"))
print(fit)
summary(fit)


print('Decison Tree Created.')
rpart.plot(fit)

print('Test Data Read.')
#range will be 1 to size-1, last element is class/target
result <- predict(fit,test, type = 'class')

print('Prediction Done.')
print(result)

printcp(fit)

opt <- which.min(fit$cptable[,'xerror'])
#get its value
cp <- fit$cptable[opt, 'CP']
pruned_model <- prune(fit,cp)
#plot tree
plot(pruned_model);text(pruned_model)
pruned_model_predict<-predict(pruned_model,test,type='class')
mean(pruned_model_predict==test$Species)

printcp(pruned_model)

mean(result==test$Species)
table(pred=result,true=test$Species)
confusionMatrix(result, test$Species)

############################################################
############### SVM ########################################
############################################################

library(e1071)
library(mlbench)
index <- 1:nrow(b)
trainIndex <- sample(1:nrow(b), 0.8 * nrow(b))
test <- b[testindex,]
train <- b[-testindex,]

mytunedsvm <- tune.svm(Species ~ ., data = train, gamma = 2^(-1:1), cost = 2^(2:4)) 
summary(mytunedsvm)

svm.model <- svm(Species ~ ., data = train,kernel = 'linear', cost = 4, gamma = 0.5,type='C-classification')
svm.pred <- predict(svm.model, test,type='C-Classification')
table(pred = svm.pred, true = test$Species)
mean(svm.pred==test$Species)
summary(svm.model)
summary(svm.pred)
print(svm.pred)
confusionMatrix(svm.pred, test$Species)

plot(svm.model, b,Petal.Width~Petal.Length,
     slice = list(Sepal.Width=4,Sepal.Length = 3))

#############################################################
###########################Naive baye########################
#############################################################
library(e1071)
library(MASS)
library(klaR)
c<-iris
split=0.80
trainIndex <- createDataPartition(c$Species, p=split, list=FALSE)
data_train <- c[ trainIndex,]
data_test <- c[-trainIndex,]
# train a naive bayes model
model <- NaiveBayes(Species~., data=data_train )
# make predictions
x_test <- data_test[,1:4]
y_test <- data_test[,5]
predictions <- predict(model, x_test[-5],y_test[5],type='raw')
# summarize results
confusionMatrix(predictions$class, y_test)
plot(model)

#########################################################################################################
###############Program to run alll the classiers on different ratio of Test and train data SPlit#########
#########################################################################################################
for(i in 1:5){
  splitting=0.9
  trainIndex <- sample(1:nrow(scaled), splitting * nrow(scaled))
  train <- scaled[trainIndex, ]
  test <- scaled[-trainIndex, ]
  
  #neural net
  
  f <- as.formula(paste("Species ~", paste(n[!n %in% "Species"], collapse = " + ")))
  nn <- neuralnet(f,data=train,hidden=c(2,7),threshold = 0.05,rep=5)
  
  pred <- compute(nn,test[,1:4])
  pred.scaled <- pred$net.result *(max(b$Species)-min(b$Species))+min(b$Species)
  real.values <- (test$Species)*(max(b$Species)-min(b$Species))+min(b$Species)
  print(i)
  print("Neural")
  MSE.nn <- sum((real.values - pred.scaled)^2)/nrow(test)
  print(MSE.nn)
  
  
  #Perceptron
  
  nn <- neuralnet(f,data=train,hidden=0,threshold = 0.0001)
  
  pred <- compute(nn,test[,1:4])
  pred.scaled <- pred$net.result *(max(b$Species)-min(b$Species))+min(b$Species)
  real.values <- (test$Species)*(max(b$Species)-min(b$Species))+min(b$Species)
  print(i)
  print("Perceptron")
  MSE.nn <- sum((real.values - pred.scaled)^2)/nrow(test)
  print(MSE.nn)
  
  ## Decision Tree
  fit <- rpart(f, data = train, method =
                 'class', parms = list(split = "information"))
  
  result <- predict(fit,test, type = 'class')
  print(i) 
  print("Decision Tree")
  print(confusionMatrix(result, test$Species))
  
  #SVM
  trainIndex <- sample(1:nrow(b), splitting * nrow(b))
  train <- b[trainIndex, ]
  test <- b[-trainIndex, ]
  svm.model <- svm(Species ~ ., data = train,kernel = 'linear', cost = 4, gamma = 0.5,type='C-classification')
  svm.pred <- predict(svm.model, test,type='C-Classification')
  print(i) 
  print("SVM")
  print(confusionMatrix(svm.pred, test$Species))
  
  #Naive
  trainIndex <- createDataPartition(c$Species, p=splitting, list=FALSE)
  data_train <- c[ trainIndex,]
  data_test <- c[-trainIndex,]
  model <- NaiveBayes(Species~., data=data_train )
  # make predictions
  x_test <- data_test[,1:4]
  y_test <- data_test[,5]
  predictions <- predict(model, x_test,type='raw')
  print(i)
  print("Naive")
  print(confusionMatrix(predictions$class, y_test))
  splitting = splitting - 0.10
  
}


