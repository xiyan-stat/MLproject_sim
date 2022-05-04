#############################################################
#                    MATH9810 Project R code                #
#     Comparison between NN and RF in classification        #      
#              Xiyan Tan, Jiajing Niu, Li He                #
#############################################################

###############################################################
#           MATH 9810 Project                                 #
#       Simulation 1: Neural Network vs Random Forest         #
###############################################################

##SIMULATION 1

##Generate data:
#- Six predictor variables (x1 to x6) and one response variable (y).
#- Correlations between predictor variables are set to 0.
#- Mean of each variable is zero, and the variance is 1.

#### load library
# Neural network & tree-based algorithm
library(neuralnet)
library(MASS)
library(glmnet)
library(nnet)
library(mvtnorm)
library(caret)
library(rpart.plot)
library(rpart)
library (randomForest)
library(ipred)
library(tidyverse)
library(matlab)

# Sigmoid function
sigmoid <- function(z){1.0/(1.0+exp(-z))}
#variance-covariance matrix
varcov <- diag(6)

#statistical population
statpop.X <- rmvnorm(10000, mean=rep(0, 6), sigma=varcov)
signal_to_noise_ratio <- 4
a1 <- c(3,3,3,-3,-3,-3)
noise <- rnorm(10000)
fx <- sigmoid(statpop.X%*%a1)
k <- sqrt(var(fx)/(signal_to_noise_ratio*var(noise)))
statpop.Y <- fx + as.vector(k)*noise
statpop.Yb <- ifelse(statpop.Y>=0.5,1,0)
statpop <- cbind(statpop.Yb,statpop.X)
colnames(statpop) <- c("y", "x1", "x2", "x3", "x4", "x5","x6")

apply(statpop,2,mean) #means of variables
round(var(statpop),2) #variances-covariances
round(cor(statpop),2) #correlations
summary(glm(y~., data=data.frame(statpop))) 

#sample
randSampleObs <- sample(10000, 2*500, replace=FALSE)
randSample <- data.frame(statpop[randSampleObs,])
colnames(randSample) <- c("y", "x1", "x2", "x3", "x4", "x5","x6")


#########################################
# Choose hidden layer and weight decays #
#########################################

#split random sample into training and validation set
index <- sample(nrow(randSample), size=.2*(nrow(randSample)), replace=FALSE)
#training data
randSampleT <-randSample[index,]
#validation data 
randSampleV <-randSample[-index,]

#glm: zero layers (check layer 0)
lm.fit <- glm(y~., data=data.frame(randSampleT),family = "binomial")
summary(lm.fit)
pr.lm <- predict(lm.fit,randSampleV)
pr.lm <- ifelse(pr.lm>=0.5,1,0)
misclass.lm <- mean(pr.lm != randSampleV$y)
misclass.lm

n <- names(randSampleT)
f <- as.formula(paste("y ~", paste(n[!n %in% "y"], collapse = " + ")))
### 1. Boxplot for number of Hidden units
hiddenunits <- seq(0,10,1)
m <- length(hiddenunits)
k <- 10
test.error <- matrix(0,nrow=m,ncol=k)
set.seed(450)
for (i in 1:m){  
  for(j in 1:k){
    # fit neural network
    nn <- neuralnet(f, data=data.frame(randSampleT), hidden=hiddenunits[i], act.fct = "logistic",linear.output = T)
    pr.nn <- compute(nn,randSampleV[,2:7])
    pr.nn <- ifelse(pr.nn$net.result>=0.5,1,0)
    test.error[i,j] <- mean(pr.nn != randSampleV$y)
  }
}

misclass.nn <- test.error

#The code for the box plot of the CV error: choose hidden units=2
boxplot(misclass.nn~hiddenunits,xlab='number of hidden units', ylab="Test error",
        col='orange',border='purple',main='Test error for NN with different hidden units',horizontal=FALSE)

### 2. Boxplot for weightdecay
decay.func <- function(a){
  k <- 10
  m <- length(a)
  test.error <- matrix(0,nrow=m,ncol=k)
  for(i in 1:k){
    # fit neural network
    nn2 <- nnet(y~., data=data.frame(randSampleT), size=2, decay=a,maxit=200)
    pr.nn2 <- predict(nn2,randSampleV[,2:7])
    pr.nn2 <- ifelse(pr.nn2>=0.5,1,0)
    test.error[,i] <- mean(pr.nn2 != randSampleV$y)
  }
  return(test.error)
}

weightdecay <- seq(0,0.15,by=0.01)
misclass2.nn <- sapply(weightdecay,decay.func)
misclass2.nn <- t(misclass2.nn)


#The code for the box plot of the CV error: choose weighted decay=0.12
boxplot(misclass2.nn~weightdecay,xlab='weight Decay Parameter', ylab="Test error",
        col='orange',border='purple',main='Test error for different decay with 2 hiddens',horizontal=FALSE)

#########################################
# Comparison among different algorithms #
#########################################

#table 1
### 1. no hidden
nn0 <- neuralnet(f, data=data.frame(randSampleT), hidden=0, act.fct = "logistic",linear.output = T)
pr.nn0 <- compute(nn0,randSampleV[,2:7])
pr.nn0 <- ifelse(pr.nn0$net.result>=0.5,1,0)
misclass.nn0 <- mean(pr.nn0 != randSampleV$y)

### 2. Single layer (no decay)
nn1 <- neuralnet(f, data=data.frame(randSampleT), hidden=2, act.fct = "logistic",linear.output = T)
plot(nn1)
pr.nn1 <- compute(nn1,randSampleV[,2:7])
pr.nn1 <- ifelse(pr.nn1$net.result>=0.5,1,0)
misclass.nn1 <- mean(pr.nn1 != randSampleV$y)

### 3. Single layer (with decay)
nn2 <- nnet(y~., data=data.frame(randSampleT), size=2, decay=0.12,maxit=200)
pr.nn2 <- predict(nn2,randSampleV[,2:7])
pr.nn2 <- ifelse(pr.nn2>=0.5,1,0)
misclass.nn2 <- mean(pr.nn2 != randSampleV$y)

### 4. Two Layers (layer1=2, layer2=2, no decay)
nn3 <- neuralnet(f, data=data.frame(randSampleT), hidden=c(2,2), act.fct = "logistic",linear.output = T)
pr.nn3 <- compute(nn3,randSampleV[,2:7])
pr.nn3 <- ifelse(pr.nn3$net.result>=0.5,1,0)
misclass.nn3 <- mean(pr.nn3 != randSampleV$y)

### 5. Decision Tree (gini split)
dt <- rpart(y ~., data = data.frame(randSampleT))
prp(dt)
pr.dt <- predict(dt, newdata = randSampleV)
pr.dt <- ifelse(pr.dt>=0.5,1,0)
misclass.dt <- mean(pr.dt != randSampleV$y)

### 6. Bagging
bag <- bagging(y ~ ., data = data.frame(randSampleT), coob=TRUE)
pr.bag<-predict(bag, newdata=randSampleV)
pr.bag <- ifelse(pr.bag>=0.5,1,0)
misclass.bag <- mean(pr.bag != randSampleV$y)

###7.Random Forest
rf <- randomForest(as.factor(y)~.,data =data.frame(randSampleT), importance = TRUE)
pr.rf<-predict(rf, newdata=randSampleV)
misclass.rf <- mean(pr.rf != randSampleV$y)

#Simulation 1 Result
res1 <- c(misclass.nn0,misclass.nn1,misclass.nn2,misclass.nn3,misclass.dt,misclass.bag,misclass.rf)
res1

###############################################################
#           MATH 9810 Project                                 #
#       Simulation 2: NN vs RF with correlation               #
###############################################################

##SIMULATION 2

##Generate data:
#- Six predictor variables (x1 to x6) and one response variable (y).
#- Correlations between predictor variables are set to 0.8,0.6,0.4,0.2,0.1.
#- Mean of each variable is zero, and the variance is 1.


# Sigmoid function
sigmoid <- function(z){1.0/(1.0+exp(-z))}
#variance-covariance matrix
varcov <- matrix(c(1, .8, .6, .4, .2, .1,
                   .8, 1, .8, .6, .4, .2,
                   .6, .8, 1, .8, .6, .4,
                   .4, .6, .8 ,1, .8, .6,
                   .2, .4, .6, .8, 1, .8,
                   .1, .2, .4, .6, .8, 1), ncol=6, byrow=TRUE)

#statistical population
statpop.X <- rmvnorm(10000, mean=rep(0, 6), sigma=varcov)
signal_to_noise_ratio <- 4
a1 <- c(3,3,3,-3,-3,-3)
fx <- sigmoid(statpop.X%*%a1)
statpop.Yb <- rbinom(10000,1,fx)
statpop <- cbind(statpop.Yb,statpop.X)
colnames(statpop) <- c("y", "x1", "x2", "x3", "x4", "x5","x6")

apply(statpop,2,mean) #means of variables
round(var(statpop),2) #variances-covariances
round(cor(statpop),2) #correlations
summary(glm(y~., data=data.frame(statpop))) 

#sample
randSampleObs <- sample(10000, 2*500, replace=FALSE)
randSample <- data.frame(statpop[randSampleObs,])
colnames(randSample) <- c("y", "x1", "x2", "x3", "x4", "x5","x6")


#########################################
# Choose hidden layer and weight decays #
#########################################

#split random sample into training and validation set
index <- sample(nrow(randSample), size=.6*(nrow(randSample)), replace=FALSE)
#training data
randSampleT <-randSample[index,]
#validation data 
randSampleV <-randSample[-index,]

n <- names(randSampleT)
f <- as.formula(paste("y ~", paste(n[!n %in% "y"], collapse = " + ")))
# choose hidden units and weights

sizes <- seq(1,10,1)
decays <- seq(0,0.15,by=0.01)
tryValues <- expand.grid(size=sizes, decay=decays)
trainData <- data.frame(randSampleT)
nrep <- 1
errorCV <- sapply(1:nrow(tryValues), function (i)
  replicate(nrep, errorest(y~., data=trainData, model=nnet, linout=TRUE,
                           size=tryValues[i,1], decay=tryValues[i,2], trace=FALSE,
                           estimator="cv",
                           est.para=list(control.errorest(k=10)))$error),simplify=FALSE)
index = which.min(errorCV)
plot(tryValues[,1],errorCV)
plot(tryValues[,2],errorCV)

#########################################
# Comparison among different algorithms #
#########################################

#table 1
### 1. no hidden
nn0 <- neuralnet(f, data=data.frame(randSampleT), hidden=0, act.fct = "logistic",linear.output = T)
pr.nn0 <- neuralnet::compute(nn0,randSampleV[,2:7])
pr.nn0 <- ifelse(pr.nn0$net.result>=0.5,1,0)
misclass.nn0 <- mean(pr.nn0 != randSampleV$y)
misclass.nn0

### 2. Single layer (no decay)
nn1 <- neuralnet(f, data=data.frame(randSampleT), hidden=3, act.fct = "logistic",linear.output = T)
#plot(nn1)
pr.nn1 <- neuralnet::compute(nn1,randSampleV[,2:7])
pr.nn1 <- ifelse(pr.nn1$net.result>=0.5,1,0)
misclass.nn1 <- mean(pr.nn1 != randSampleV$y)
misclass.nn1

### 3. Single layer (with decay)
nn2 <- nnet(y~., data=data.frame(randSampleT), size=3, decay=0.03,maxit=200)
pr.nn2 <- predict(nn2,randSampleV[,2:7])
pr.nn2 <- ifelse(pr.nn2>=0.5,1,0)
misclass.nn2 <- mean(pr.nn2 != randSampleV$y)
misclass.nn2

### 4. Two Layers (layer1=2, layer2=2, no decay)
nn3 <- neuralnet(f, data=data.frame(randSampleT), hidden=c(3,2), act.fct = "logistic",linear.output = T)
pr.nn3 <- neuralnet::compute(nn3,randSampleV[,2:7])
pr.nn3 <- ifelse(pr.nn3$net.result>=0.5,1,0)
misclass.nn3 <- mean(pr.nn3 != randSampleV$y)
misclass.nn3

### 5. Decision Tree (gini split)
dt <- rpart(y ~., data = data.frame(randSampleT))
prp(dt)
pr.dt <- predict(dt, newdata = randSampleV)
pr.dt <- ifelse(pr.dt>=0.5,1,0)
misclass.dt <- mean(pr.dt != randSampleV$y)

### 6. Bagging
bag <- bagging(y ~ ., data = data.frame(randSampleT), coob=TRUE)
pr.bag<-predict(bag, newdata=randSampleV)
pr.bag <- ifelse(pr.bag>=0.5,1,0)
misclass.bag <- mean(pr.bag != randSampleV$y)

###7.Random Forest
rf <- randomForest(as.factor(y)~.,data =data.frame(randSampleT), importance = TRUE)
pr.rf<-predict(rf, newdata=randSampleV)
misclass.rf <- mean(pr.rf != randSampleV$y)

#Simulation 2 Result
res2 <- c(misclass.nn0,misclass.nn1,misclass.nn2,misclass.nn3,misclass.dt,misclass.bag,misclass.rf)
res2

###############################################################
#           MATH 9810 Project                                 #
#       Simulation 3: NN vs RF nonlinear (indep)              #
###############################################################

##SIMULATION 3

##Generate data:
#- Six predictor variables (x1 to x6) and one response variable (y).
#- Nonlinear model
#- Mean of each variable is zero, and the variance is 1.

# Sigmoid function
sigmoid <- function(z){1.0/(1.0+exp(-z))}
#variance-covariance matrix
varcov <- diag(6)

#statistical population
statpop.X <- rmvnorm(10000, mean=rep(0, 6), sigma=varcov)
statpop.X <- data.frame(statpop.X)
statpop.X <- model.matrix(~(X1+X2)^2+(X3+X4)^2+X5+X6-1,statpop.X)
statpop.X <- as.matrix(cbind(statpop.X,statpop.X[,5]^2,statpop.X[,6]^4))
a1 <- c(3,3,3,-3,-3,-3,3,3,-3,-3)
fx <- sigmoid(statpop.X%*%a1)
statpop.Yb <- rbinom(10000,1,fx)
statpop <- cbind(statpop.Yb,statpop.X)
colnames(statpop) <- c("y", "x1", "x2", "x3", "x4", "x5","x6","x1x2","x3x4","x52","x64")

apply(statpop,2,mean) #means of variables
round(var(statpop),2) #variances-covariances
round(cor(statpop),2) #correlations
summary(glm(y~., data=data.frame(statpop))) 

#sample
randSampleObs <- sample(10000, 2*500, replace=FALSE)
randSample <- data.frame(statpop[randSampleObs,])
colnames(randSample) <- c("y", "x1", "x2", "x3", "x4", "x5","x6","x1x2","x3x4","x52","x64")


#########################################
# Choose hidden layer and weight decays #
#########################################

#split random sample into training and validation set
index <- sample(nrow(randSample), size=.6*(nrow(randSample)), replace=FALSE)
#training data
randSampleT <-randSample[index,]
#validation data 
randSampleV <-randSample[-index,]

n <- names(randSampleT)
f <- as.formula(paste("y ~", paste(n[!n %in% "y"], collapse = " + ")))

# choose hidden units and weights
library(ipred)
sizes <- seq(1,10,1)
decays <- seq(0,0.15,by=0.01)
tryValues <- expand.grid(size=sizes, decay=decays)
trainData <- data.frame(randSampleT)
nrep <- 1
errorCV <- sapply(1:nrow(tryValues), function (i)
  replicate(nrep, errorest(f, data=trainData, model=nnet, linout=TRUE,
                           size=tryValues[i,1], decay=tryValues[i,2], trace=FALSE,
                           estimator="cv",
                           est.para=list(control.errorest(k=10)))$error),simplify=FALSE)
index = which.min(errorCV)
plot(tryValues[,1],errorCV)
plot(tryValues[,2],errorCV)

#########################################
# Comparison among different algorithms #
#########################################

#table 1
### 1. no hidden
nn0 <- neuralnet(f, data=data.frame(randSampleT), hidden=0, act.fct = "logistic",linear.output = T)
pr.nn0 <- neuralnet::compute(nn0,randSampleV[,2:11])
pr.nn0 <- ifelse(pr.nn0$net.result>=0.5,1,0)
misclass.nn0 <- mean(pr.nn0 != randSampleV$y)

### 2. Single layer (no decay)
nn1 <- neuralnet(f, data=data.frame(randSampleT), hidden=3, act.fct = "logistic",linear.output = T)
#plot(nn1)
pr.nn1 <- neuralnet::compute(nn1,randSampleV[,2:11])
pr.nn1 <- ifelse(pr.nn1$net.result>=0.5,1,0)
misclass.nn1 <- mean(pr.nn1 != randSampleV$y)
misclass.nn1

### 3. Single layer (with decay)
nn2 <- nnet(y~., data=data.frame(randSampleT), size=3, decay=0.11,maxit=200)
pr.nn2 <- predict(nn2,randSampleV[,2:11])
pr.nn2 <- ifelse(pr.nn2>=0.5,1,0)
misclass.nn2 <- mean(pr.nn2 != randSampleV$y)

### 4. Two Layers (layer1=2, layer2=2, no decay)
nn3 <- neuralnet(f, data=data.frame(randSampleT), hidden=c(3,2), act.fct = "logistic",linear.output = T)
pr.nn3 <- neuralnet::compute(nn3,randSampleV[,2:11])
pr.nn3 <- ifelse(pr.nn3$net.result>=0.5,1,0)
misclass.nn3 <- mean(pr.nn3 != randSampleV$y)

### 5. Decision Tree (gini split)
dt <- rpart(y ~., data = data.frame(randSampleT))
#prp(dt)
pr.dt <- predict(dt, newdata = randSampleV)
pr.dt <- ifelse(pr.dt>=0.5,1,0)
misclass.dt <- mean(pr.dt != randSampleV$y)

### 6. Bagging
bag <- bagging(y ~ ., data = data.frame(randSampleT), coob=TRUE)
pr.bag<-predict(bag, newdata=randSampleV)
pr.bag <- ifelse(pr.bag>=0.5,1,0)
misclass.bag <- mean(pr.bag != randSampleV$y)

###7.Random Forest
rf <- randomForest( as.factor(y)~.,data =data.frame(randSampleT), importance = TRUE)
pr.rf<-predict(rf, newdata=randSampleV)
misclass.rf <- mean(pr.rf != randSampleV$y)

# Simluation 3 Results
res3 <- c(misclass.nn0,misclass.nn1,misclass.nn2,misclass.nn3,misclass.dt,misclass.bag,misclass.rf)
res3

# plot all results
dat <- cbind(res1,res2,res3)
matplot(dat, type = c("b"),pch=1,col = 1:3) #plot
legend("topleft", legend = 1:3, col=1:3, pch=1)


#################################################
#           MATH 9810 Project                   #
#       Application 1: Zipcode Data             #
#################################################

# read train and test zipcpde data
# read zip.train
ziptrain <- read.table(file.choose())
n1 <- dim(ziptrain)[1]
# read zip.test
ziptest <- read.table(file.choose())
n2 <- dim(ziptest)[1]

# standarized (already normalized)
zipdata <- rbind(ziptrain,ziptest)
X <- zipdata[,-1]
xb <- apply(X,2,mean)
sx <- apply(X,2,sd)
Xs <- t((t(X)-xb)/sx)
zipdata.sd <- cbind(zipdata[,1],Xs)
ziptrains <- data.frame(zipdata.sd[1:n1,])
ziptests <- data.frame(zipdata.sd[(n1+1):(n1+n2),])

# Plot the 16*16 grayscale images of handwritten numbers
#plot train image examples
library(imager)
im <- matrix(as.numeric(ziptrain[2,2:257]), nrow = 16, ncol = 16)
image(t(apply(-im,1,rev)),col=gray((0:32)/32))
ziptrain[2,1]
im <- matrix(as.numeric(ziptrain[1,2:257]), nrow = 16, ncol = 16)
image(t(apply(-im,1,rev)),col=gray((0:32)/32))
# correct value
ziptrain[1,1]

#plot selected test image examples: 0-9
library(imager)
kk <- c(6,17,13,3,14,37,4,31,29,1)
for (i in kk){
  im.test <- matrix(as.numeric(ziptest[i,2:257]), nrow = 16, ncol = 16)
  image(t(apply(-im.test,1,rev)),col=gray((0:32)/32))
}
#correct value
ziptest[kk,1]

#########################################
# Comparison among different algorithms #
#########################################
n <- names(ziptrains)
f <- as.formula(paste("V1 ~", paste(n[!n %in% "V1"], collapse = " + ")))

#table 2
### 1. no hidden (multilogistic regression)

tic()
nn0 <- nnet::multinom(V1 ~., data = data.frame(ziptrains),MaxNWts=3000)
pr.nn0 <- nn0 %>% predict(ziptests)
accuracy.nn0 <- mean(pr.nn0[1:20] == ziptests$V1[1:20])
accuracy.nn0
toc()

### 2. Single layer (no decay)
library(nnet)
tic()
nn1 <- nnet(as.factor(V1)~., data=ziptrain, size=12, MaxNWts=6000)
pre.nn1 <-predict(nn1,ziptest,type="class")
accuracy.nn1 <- mean(pre.nn1 == ziptest$V1)
accuracy.nn1
toc()

### 3. Single layer (with decay)
tic()
nn2 <- nnet(as.factor(V1)~., data=ziptrain, size=12, decay=10, MaxNWts=6000)
pre.nn2 <-predict(nn2,ziptest,type="class")
accuracy.nn2 <- mean(pre.nn2 == ziptest$V1)
accuracy.nn2
toc()

# ### 4. Two Layers (layer1=2, layer2=2, no decay)
# nn3 <- neuralnet(f, data=data.frame(ziptrain), hidden=c(20,12), act.fct = "logistic",linear.output = F,lifesign = "minimal")
# pr.nn3 <- neuralnet::compute(nn3,ziptest[,2:257])$net.result
# pr.nn3 <- max.col(pr.nn3)
# accuracy.nn3 <- mean(pr.nn3 == ziptest$V1)

### 5. Decision Tree (gini split)
library(rpart)
library(rpart.plot)
tic()
dt <-rpart(as.character(V1)~.,data=ziptrain)
#prp(dt)
pr.dt <- predict(dt,ziptest,type="class")
accuracy.dt <- mean(as.numeric(as.character(pr.dt))==ziptest[,1])
accuracy.dt
toc()

###6.Random Forest
library(randomForest)
rf <- randomForest((V1)~.,data=ziptrain,type="class")
plot(rf,main='Error vs No. of trees plot: Base Model')
# number of trees 150
tic()
rf1 <- randomForest((V1)~.,data=ziptrain,type="class",keep.forest=T, ntree=150)
pre.df <- predict(rf1, ziptest,type="class")
pre.df <- as.integer(pre.df)
accuracy.rf <- mean(pre.df==ziptest$V1)
accuracy.rf
toc()
# importance of RF
imp.rf <- importance(rf)
vars <- dimnames(imp.rf)[[1]]
imp <- data.frame(vars=vars,imp.rf=as.numeric(imp.rf[,1]))
imp.rf <- imp.rf[order(imp.rf$imp.rf,decreasing=T),]
varImpPlot(rf,main='Variable Importance Plot: Base Model')


# Results of Zip code data
res4 <- c(accuracy.nn0,accuracy.nn1,accuracy.nn2,accuracy.dt,accuracy.rf)
res4
#[1] 0.8500000 0.8834081 0.9098156 0.7249626 0.4539113

# plot some test examples from 1:40
image <- matrix(0,nrow=6,ncol=40)
for (i in 1:40){
  image[,i] <- c(ziptest[i,1],as.character(pr.nn0[i]),pre.nn1[i],pre.nn2[i],as.character(pr.dt[i]),pre.df[i])
}
image

# multiple times to get correct error
result1 = matrix(0,nrow=10,ncol=5)
for (j in 1:10){
  tic()
  nn0 <- nnet::multinom(V1 ~., data = data.frame(ziptrains),MaxNWts=3000)
  pr.nn0 <- nn0 %>% predict(ziptests)
  accuracy.nn0 <- mean(pr.nn0[1:20] == ziptests$V1[1:20])
  accuracy.nn0
  toc()
  tic()
  nn1 <- nnet(as.factor(V1)~., data=ziptrain, size=12, MaxNWts=6000)
  pre.nn1 <-predict(nn1,ziptest,type="class")
  accuracy.nn1 <- mean(pre.nn1 == ziptest$V1)
  accuracy.nn1
  toc()
  tic()
  nn2 <- nnet(as.factor(V1)~., data=ziptrain, size=12, decay=10, MaxNWts=6000)
  pre.nn2 <-predict(nn2,ziptest,type="class")
  accuracy.nn2 <- mean(pre.nn2 == ziptest$V1)
  accuracy.nn2
  toc()
  tic()
  dt <-rpart(as.character(V1)~.,data=ziptrain)
  #prp(dt)
  pr.dt <- predict(dt,ziptest,type="class")
  accuracy.dt <- mean(as.numeric(as.character(pr.dt))==ziptest[,1])
  accuracy.dt
  toc()
  tic()
  rf1 <- randomForest((V1)~.,data=ziptrain,type="class",keep.forest=T, ntree=150)
  pre.df <- predict(rf1, ziptest,type="class")
  pre.df <- as.integer(pre.df)
  accuracy.rf <- mean(pre.df==ziptest$V1)
  accuracy.rf
  toc()
  result1[j,] <- c(accuracy.nn0,accuracy.nn1,accuracy.nn2,accuracy.dt,accuracy.rf)
}

# optimal correct rate for each algorithm
apply(result1,2,max)

# > result1
# [,1]      [,2]      [,3]      [,4]      [,5]
# [1,] 0.85 0.8873941 0.9088191 0.7249626 0.4598904
# [2,] 0.85 0.8858994 0.9133034 0.7249626 0.4564026
# [3,] 0.85 0.8749377 0.9162930 0.7249626 0.4573991
# [4,] 0.85 0.8858994 0.9157947 0.7249626 0.4499253
# [5,] 0.85 0.8903837 0.9128052 0.7249626 0.4499253
# [6,] 0.85 0.8844046 0.9138017 0.7249626 0.4529148
# [7,] 0.85 0.8779273 0.9152965 0.7249626 0.4534131
# [8,] 0.85 0.8903837 0.9068261 0.7249626 0.4559043
# [9,] 0.85 0.8804185 0.9128052 0.7249626 0.4573991
# [10,] 0.85 0.8819133 0.9138017 0.7249626 0.4658695

#time elapsed
#NN average tuning time: 104sec
# NN0     NN1     NN2    DT    RF
# 18.226	22.894	25.575	4.12	71.16
# 18.92	25.83	25.876	4.28	74.999
# 19.395	24.181	27.6	4.339	74.793
# 18.856	25.063	27.205	4.397	74.951
# 19.314	25.043	27.969	4.328	74.661
# 19.422	24.563	28.973	4.299	74.506
# 18.804	23.986	27.45	4.262	73.906
# 18.844	24.411	27.424	4.269	74.942
# 18.949	24.288	26.669	4.265	74.563
# 18.751	23.531	26.793	4.209	73.795
# mean 
# 18.9481	24.379	27.1534	4.2768	74.2276
# standard deviation
# 0.358908051	0.828338095	0.9903471	0.075309436	1.155998097


#################################################
#           MATH 9810 Project                   #
#       Application 2: Heart Diseases Data      #
#################################################

# read the heart data
# read Heart.txt
heart <- read.table(file.choose(), sep=",", header=TRUE)
Y.heart <- as.numeric(heart[,11])
X.heart <- cbind(heart[,2:4],heart[,6]=="Present",heart[,8:10])
X.heart <- as.matrix(X.heart)

#standarized
xb <- apply(X.heart,2,mean)
sx <- apply(X.heart,2,sd)
Xs.heart <- t((t(X.heart)-xb)/sx)
data.heart <- data.frame(cbind(Y.heart, Xs.heart))

#random select train and test set
#split random sample into training and validation set
index <- sample(nrow(Xs.heart), size=.6*(nrow(Xs.heart)), replace=FALSE)
#training data 60%
heart.train <- data.frame(cbind(Y.heart[index],Xs.heart[index,]))
#validation data 40%
heart.test <- data.frame(cbind(Y.heart[-index],Xs.heart[-index,]))


#########################################
# Comparison among different algorithms #
#########################################
# choose hidden units and weights

sizes <- seq(1,10,1)
decays <- seq(0,1,by=0.01)
tryValues <- expand.grid(size=sizes, decay=decays)
trainData <- data.frame(data.heart)
nrep <- 1
errorCV <- sapply(1:nrow(tryValues), function (i)
  replicate(nrep, errorest(Y.heart~., data=trainData, model=nnet, linout=TRUE,
                           size=tryValues[i,1], decay=tryValues[i,2], trace=FALSE,
                           estimator="cv",
                           est.para=list(control.errorest(k=10)))$error),simplify=FALSE)
plot(tryValues[,1],errorCV)
plot(tryValues[,2],errorCV)

n <- names(heart.train)
f <- as.formula(paste("V1~", paste(n[!n %in% "V1"], collapse = " + ")))
#table 1
### 1. no hidden
tic()
nn0 <- nnet::multinom(V1 ~., data = data.frame(heart.train),MaxNWts=3000)
pr.nn0 <- nn0 %>% predict(heart.test)
acc.nn0<- mean(pr.nn0 == heart.test$V1)
acc.nn0
toc()

### 2. Single layer (no decay)
tic()
nn1 <- nnet(as.factor(V1)~., data=heart.train, size=9, MaxNWts=3000)
pre.nn1 <-predict(nn1,heart.test,type="class")
acc.nn1 <- mean(pre.nn1 == heart.test$V1)
acc.nn1
toc()

### 3. Single layer (with decay)
tic()
nn2 <- nnet(as.factor(V1)~., data=heart.train, size=9, decay=1, MaxNWts=3000)
pre.nn2 <-predict(nn2,heart.test,type="class")
acc.nn2 <- mean(pre.nn2 == heart.test$V1)
acc.nn2
toc()

### 4. Two Layers (layer1=2, layer2=1, no decay)
tic()
nn3 <- neuralnet(V1~., data=data.frame(heart.train), hidden=c(9,5), act.fct = "logistic",linear.output = T)
pr.nn3 <- neuralnet::compute(nn3,heart.test[,2:8])
pr.nn3 <- ifelse(pr.nn3$net.result>=0.5,1,0)
acc.nn3 <- mean(pr.nn3 == heart.test[,1])
acc.nn3
toc()

### 5. Decision Tree (gini split)
tic()
dt <- rpart(as.character(V1)~.,data=heart.train)
#prp(dt)
pr.dt <- predict(dt,heart.test,type="class")
acc.dt <- mean(as.numeric(as.character(pr.dt))==heart.test[,1])
acc.dt
toc()

### 6. Bagging
tic()
bag <- bagging(V1 ~ ., data = data.frame(heart.train), coob=TRUE)
pr.bag<-predict(bag, newdata=heart.test)
pr.bag <- ifelse(pr.bag>=0.5,1,0)
acc.bag <- mean(pr.bag == heart.test[,1])
acc.bag
toc

###7.Random Forest
tic()
rf <- randomForest( as.factor(V1)~.,data =data.frame(heart.train), importance = TRUE)
pr.rf<-predict(rf, newdata=heart.test)
acc.rf <- mean(pr.rf == heart.test[,1])
acc.rf
toc()

#method2
rf <- randomForest((V1)~.,data=heart.train,type="class")
plot(rf,main='Error vs No. of trees plot: Base Model')
# number of trees 200
tic()
rf1 <- randomForest((V1)~.,data=heart.train,type="class",keep.forest=T, ntree=200)
pre.df <- predict(rf1, heart.test,type="class")
pre.df <- as.integer(pre.df)
acc1.rf <- mean(pre.df==heart.test$V1)
acc1.rf
toc()

# Results of heart diseases data
res5 <- c(acc.nn0,acc.nn1,acc.nn2,acc.nn3,acc.dt,acc.bag,acc.rf)
res5

###########################################################
#           Sigmoid Function Plot                         #
###########################################################

v <- seq(-10,10,1)
s  <- c(1,1/2,10)
f1 <- 1/(1+exp(-v))
f2 <- 1/(1+exp(-0.5*v))
f3 <- 1/(1+exp(-10*v))
#plot sigmoid function
plot(v,f1,type="l",lwd=2,lty=1,col=2, ylab=expression(1/1+e^(-v)))
lines(v,f2,lty=2,col=3,lwd=2)
lines(v,f3,lty=3,col=4,lwd=2)

