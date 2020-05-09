#Introduction:
  #This project is an analysis on Boston dataset, Goal is building a classification 
#model that Looks for the best 3 models that predict if a suburb in Boston has a crime rate below or above the median (variable crim).
  
#IMPORTING NECESSARY LIBRARIES
#I Started by importing the the folowing packages that would help in the analysis. The library(ISLR) provide the collection of data-sets used in the book ’An Introduction to Statistical Learning with Applications in R. 
#The library(MASS) provides the functions and datasets used in "Modern Applied Statistics with S. 
#The library(MASS) contains the Boston dataset to be analysed.

require(knitr)
library(ISLR)
library(MASS)
library(caret)
data(Boston)
attach(Boston)
library(xgboost)
library(tidyverse)
library(car)
library(doMC)
help(Boston)
#WORKFLOW
#As classification deals with classes, I added a class column called "resp" this represents the crime rate below or above the median (Boston$crim), where no represents crim below  median and 1 represents crim above median of variable(crim).
#The "resp" would help to predict if a suburb has a crime rate below or above the median (variable crim).

#VISUALISATION
summary(Boston)
#chad and rad are categorical data charles river dummy var and index of radial highway accesibility.
ftable(chas~rad,data=Boston)
#Histogram to show the crime rate distribution
hist(Boston$crim,col="green",main="CRIME RATE DISTRIBUTION")


#removing the outliers
calcul.mad <- function(x) {
  mad <- median(abs(x-median(x, na.rm=TRUE))) 
  mad}
Sum_f1 <- summarise_each(Boston,funs(median, calcul.mad))
n <-  2*ncol(Boston)
dl <- reshape(Sum_f1, idvar='id', direction='long', sep="_", 
              varying=split(seq(n), as.numeric(gl(n,n/2,n))))


uper.interval <- function(x,y) {
  up.inter <- median(x, na.rm=TRUE)+5*(y) 
  up.inter}
lower.interval <- function(x,y) {
  low.inter <- median(x, na.rm=TRUE)-5*(y)
  low.inter}

up_data <- mapply(uper.interval, dl[,2], dl[,3])
low_data <- mapply(lower.interval, dl[,2], dl[,3])

data_f1 <- Boston

functionData <- function(x,h,l) {
  out <- ifelse(x > h, h, ifelse(x < l, l, x))
  out}

data_f1[] <- Map(functionData, Boston, up_data, low_data)
data_f1[]
summary(data_f1[])
boxplot(data_f1[])#shows that all the outliers have been removed
class(data_f1[])
dim(data_f1[])
#drop na columns 
keeps= c("crim","indus","nox","rm","age","dis","rad","tax","ptratio","black","lstat","medv")
results=data_f1[][keeps]
summary(results)
results= na.omit(results)
dim(results)


# Create response variable
results$resp <- "No"
results$resp[crim > median(crim)] <- 'Yes'
results$resp <-factor(results$resp)
table(results$resp)
# Drop old crim variable
results <- results[-drop(1)]
summary(results)
dim(results)

#Standardization
#Im using methods that come with predefined functions for scaling.

#splitting the data
set.seed(1) #to avoid random data
sample <- sample.int(n = nrow(results), size = floor(0.5*nrow(results)), replace = F)
train <- (results[sample, ])
test  <- (results[-sample, ])
dim(train)#making sure of the split
dim(test)


#DATA ANALYSIS
 #highly correlated variables,these would not help our model predict well and ideally should be dropped.
Cor <- cor(train[,-12])
Cor
highCor <- findCorrelation(Cor, cutoff = 0.75)
highCor
#suggests we drop 2,7 and 10 nox,tax and lstat
train_cor <- train[,-drop(c(2,7,10))]
test_cor <- test[,-drop(c(2,7,10))]
dim(train_cor)
nzv <- nearZeroVar(train, saveMetrics = TRUE)#no near zero variance.
nzv

#FEATURE SELECTION TECHNIQUES AND BUILDING THE MODELS
#KNN MODEL 1
#using varables highcor suggested dropping to build a knn model
knnGrid <- expand.grid(.k=c(2))
# Use k = 2, since we expect 2 classes
set.seed(1)
KNNMODEL <- train(x=train_cor[,-9], method='knn',
             y=train_cor$resp, 
             preProcess=c('center', 'scale'), 
             tuneGrid = knnGrid)
KNNMODEL
#Accuracy   Kappa    
#0.8577  0.7155


#PREDICT USING CONFUSION MATRIX
confusionMatrix(predict(KNNMODEL, test_cor[,-9]), test_cor$resp)
#Accuracy : 0.8893  #Kappa : 0.7787 #i suspect overfitting because my test set gives a better accuracy than the train set

# CHECKING VARIABLE IMPORTANCE
z=varImp(KNNMODEL)
z
#indus most important and rm least important.
plot(z)#the higher the vbariable in the graph, the greater the importance.

#Using pricipal component analysis to fit knn a model
set.seed(1)
KNNMODEL2 <- train(x=train[,-12], method='knn',
             y=train$resp, 
             preProcess=c('center', 'scale', 'pca'), 
             tuneGrid = knnGrid)
KNNMODEL2
# Accuracy   Kappa    
#0.8414  0.6819

confusionMatrix(predict(KNNMODEL2, test[,-12]), test$resp)
#Accuracy : 0.8854 Kappa : 0.7708 

#performs better than the cor suggested knn model.

#LOGISTIC REGRESSION MODEL
logMODEL <- train(resp~., data=train, 
               method='glm', family=binomial(link='logit'),
               preProcess=c('scale', 'center'))

summary(logMODEL)
#suggests we drop indus,rm,tax,and lstat based on the pvalues being above 0.05

plot(logMODEL$finalModel,)

confusionMatrix(predict(logMODEL, test[,-12]), test$resp)
#Accuracy : 0.9012 Kappa : 0.8024
#performs better than the knn models

#REMOVING THE NON SIGNIFICANT VARIABLES AND BUILDING A NEW LOGISTIC REGRESSION MODEL
set.seed(1)
logMODEL2 <- train(resp ~ nox+age+dis+rad+ptratio+black+medv,
               data=train, 
               method='glm', family=binomial(link='logit'),
               preProcess=c('scale', 'center'))
summary(logMODEL2)

plot(logMODEL2$finalModel, which=1)

#PREDICTION
confusionMatrix(predict(logMODEL, test[,-12]), test$resp)
#Accuracy : 0.9012,Kappa : 0.8024
#confirmed that this gave me the same accuracy as the first model previous model.

#GRADIENT BOOSTING MODEL
#For this model,I used xgboost from caret workflow to automatically adjust the model parameter values, and fit the final best boosted tree that explains the best our data.
#trControl was used to set up 10-fold cross validation
set.seed(1)
GBMODEL =train(resp~., data=train,
               method="xgbTree",
               preProcess=c('scale','center'),
               trControl = trainControl("cv", number = 10)
)
GBMODEL
#Accuracy : 0.9209 Kappa : 0.8417
#at round 50, accuracy was speak.
confusionMatrix(predict(GBMODEL, test[,-12]), test$resp)
#Accuracy : 0.913  Kappa : 0.8262


#LDA MODEL
set.seed(1)
LDA <- train(resp~., data=train_cor,
             method='lda', 
             preProcess=c('scale', 'center'))
LDA

# Accuracy   Kappa   
#0.8192 0.6379

#predict
confusionMatrix(test_cor$resp, predict(LDA, test_cor[,-9]))
#Accuracy : 0.8419 Kappa : 0.684
#least accurate so far..

#QDA MODEL
set.seed(1)
QDA <- train(resp~., data=train_cor,
             method='qda', 
             preProcess=c('scale', 'center'))
QDA
#Accuracy   Kappa    
#0.8353  0.6697

confusionMatrix(test_cor$resp, predict(QDA, test_cor[,-9]))
#Accuracy : 0.8379  Kappa : 0.6761

#slightly better than lda model accuracy.

#RANDOM FOREST MODEL
set.seed(1)
rfmodel <- train(resp~., data=train_cor,
             method='rf', 
             preProcess=c('scale', 'center'))
rfmodel

#mtry selected at 7 0r 12 predictors which is all if the predictors.
#Accuracy=0.9143,   Kappa=0.8273

confusionMatrix(test$resp, predict(rfmodel, test_cor[,-9]))

#Accuracy=0.9249,   Kappa=0.8499


#DECISION TREES MODEL
treemodel<- train(resp~., data=train_cor,
             method='rpart', 
             preProcess=c('scale', 'center'))
treemodel
#Accuracy=0.8583,   Kappa=0.7156

confusionMatrix(test$resp, predict(treemodel, test_cor[,-9]))
#Accuracy=0.8261,   Kappa=0.6524

#SUMMARY
#best three models that predict the observations are 
#1. Random forest model,accuracy = 92.49%
#2.Gradient boosting model with an accuracy of 91.3%
#3.Logistic regression model wiith a test accuracy of 90.02%

