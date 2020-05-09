
#INTRODUCTION
#GOAL-predicting median house prices(medv) in a Boston #suburb.Explore regression techniques choose best three model and explain.
#loading necessary libraries and our boston data from mass library

library(MASS)
library(ISLR)
library(corrplot)
library(ggplot2)
library(dplyr)
library("caret")
library(tidyverse)
library(rpart)
library(glmnet)
library(ModelMetrics)
library ("gbm")
library(randomForest)
library(plotly)
data(Boston)
attach(Boston)

 

# EXPLORATORY ANALYSIS

summary(Boston)#getting the statistical information from the dataset,no missing values
any(is.na(Boston))
boxplot(Boston)#boxplot - variation in the values of various variables present in the dataset
#outliers in the variables crime, zn,chas,dis,ptratio,black and  lstat
plot_ly(data = Boston, x = ~medv, type = "histogram")# has a slight right skew                                                                               1000), NA)
#Highest variability is observed in the full-value property tax rates
#outliers observed in the variables crime, zn,chas,dis,ptratio,black and  lstat

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

### Scatter plot of dependent variables vs Median Value (medv) 
Boston %>%
  gather(key, val, -medv) %>%
  ggplot(aes(x = val, y = medv)) +
  geom_point() +
  stat_smooth(method = "lm", se = TRUE, col = "red") +
  facet_wrap(~key, scales = "free") +
  theme_gray() +
  ggtitle("Scatter plot of dependent variables vs Median Value (medv)") 

### Correlation -checking for patterns
cor(results)
corr_medv <- cor(results$medv, results[,-14], use = "pairwise.complete.obs")

corr_medv <- t(corr_medv)
View(corr_medv)
corr_matrix <- cor(results, use = "pairwise.complete.obs")
corrplot(corr_matrix, type = "lower", order = "hclust", 
         tl.col = "black", tl.srt = 45) #blue-negative,red-positive,colour intensity indicates the strength of the coefficient

#Observation-Strong negative correlation between percentage of lower status of population and median house price. 
#Strong positive correlation in the number of rooms and median price of the house.
#high tax reduces the median price of the house.
#variables with correlated values above 0.75 will be removed.
 #dealing with multicollinearity -this is because predictors that correlated to themselves or to the response variable
#are going to be redundant in the model-this makes it difficult to know which variable is actually contributing more in the model.
#statistically, multi collinearity makes some  var insignificant when they should be because it increases the standard error.

#Correlation matrix between feature variables and target variable
cor(results,results$medv) 
#correlation matrix measures the dependence between variables
#Correlations that are closer to +1 and-1 predict the future data more accurately
#0 means it wont be predictive atall

#remove highly correlated variables. set cut off at 0.75
#I basically just set the upper triangle to be zero and then remove any cols
#that have values over 0.75.
tmp <- cor(results)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0
data.new <- results[,!apply(tmp,2,function(x) any(abs(x) > 0.75))]
head(data.new)
dim(data.new)

#scaling 
Boston.std = scale(data.new)
Boston.std
summary(Boston.std)
Boston.std[complete.cases(Boston.std),]

#splitting the data
set.seed(1) #to avoid random data
sample <- sample.int(n = nrow(Boston.std), size = floor(0.5*nrow(Boston.std)), replace = F)
train <- as.data.frame(Boston.std[sample, ])
test  <- as.data.frame(Boston.std[-sample, ])
dim(train)#making sure of the split
dim(test)
class(train)
summary(train)


#BUILDING THE MODELS
#MODEL1
#fitting a linear model using the surviving variables to do more feature selection

set.seed(1)
linearmodel=lm(medv~., data=train)
summary(linearmodel)

#rse=0.5585,Adjusted R2=0.6629
#dis,black suggested to be dropped based on their pvalues being less than 0.05 and tvalue less than 2

#predict
set.seed(1)
pred.lm <- predict(linearmodel, newdata = test)
# mean squared error
rmse.lm <- sqrt(sum((pred.lm - test$medv)^2)/
  length(test$medv))

c(RMSE = rmse.lm, R2 = summary(linearmodel)$r.squared) 
#RMSE        R2 
#0.6519    0.6696 


#dropping dis and black variables for next modelfit-LINEAR MODEL2
set.seed(1)
linearmodel2=lm(medv~+rm+tax+ptratio ,data=train)
summary(linearmodel2)
#rse=0.5594 adjr2 =0.6618



#predict with test data
set.seed(1)
pred.linearmodel2 <- predict(linearmodel2,test)
pred.linearmodel2
# mean squared error
rmse.linearmodel2=sqrt(mse.linearmodel2 <- sum((pred.linearmodel2 - test$medv)^2)/
  length(test$medv))
c(RMSE = rmse.linearmodel2, R2 = summary(linearmodel2)$r.squared)
#RMSE        R2 
#0.6514    0.6658 
#very minute reduction in rmse and slight decrease in R2

#BACKWARD ELIMINATION-Technique to select and drop insignificant variables,starts with the full least squares model containing
#all given predictors, it iteratively removes the least useful predictor 

beselect= step(linearmodel,direction='backward')#confirmed all but black -Pvalue above 0.05
summary(beselect)
#this confirms also, the "black" i dropped in the second model that yielded a better error.

#MODEL3
set.seed(1)
linearmodel3=lm(medv~+dis+rm+tax+ptratio ,data=train)
summary(linearmodel3)

#Dropped suggested variable black,didnt perform better

# Plot of predicted price vs actual price of the best linear model 2
plot(pred.linearmodel2,test$medv, xlab = "Predicted Price", ylab = "Actual Price")

# Diagnostics plots
layout(matrix(c(1,2,3,4),2,2))
plot(linearmodel2)


#RANDOM FOREST MODEL
set.seed(1)
rfmodel <- randomForest(formula = medv ~ ., data = train)

#predict
set.seed(1)
pred.rf <- predict(rfmodel, test)

rmse.rf <- sqrt(sum(((pred.rf) - test$medv)^2)/
                  length(test$medv))
c(RMSE = rmse.rf, pseudoR2 = mean(rfmodel$rsq))
#RMSE  pseudoR2 
#0.5875 0.7479
#random forest gives a better model than the first two..decreased error,increased R^2


#plot of predicted price versus actual price
plot(pred.rf,test$medv, xlab = "Predicted Price", ylab = "Actual Price", pch = 3)




#RANDOM FOREST WITH BAGGING(ensemble)
set.seed(1)
bag.rf=randomForest(medv~.,train,mtry=5,importance=TRUE) #setting mtry=5 means all 5 predictors are considered for each split of the tree i.e bagging should be done.
bag.rf
#mse=0.2299,%var explained=75.08
#from my exploration,random forest method gives a better model than the random forest alg i did with bagging so i go with the random forest and drop the bagging

#PREDICT
set.seed(1)
pred.bag.rf <- predict(bag.rf, test)

rmse.bag.rf <- sqrt(sum(((pred.bag.rf) - test$medv)^2)/
  length(test$medv))
c(RMSE = rmse.bag.rf, pseudoR2 = mean(bag.rf$rsq))
#    RMSE  pseudoR2 
#0.6283915 0.7488696 
#observation-bagging didnt improve the model.

#RIDGE REGRESSION
#grid range for lambda
lambda <- 10^seq(-3, 3, length = 100)
# Build the model
set.seed(1)
ridge <- train(
  medv ~., data = train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda)
)

# Model coefficients
coef(ridge$finalModel, ridge$bestTune$lambda)
# Make predictions
predictions <- ridge %>% predict(test)
# Model prediction performance
data.frame(
  Rsquare = R2(predictions, test$medv)
)
rmse.ridge <- sqrt(sum(((predictions) - test$medv)^2)/
  length(test$medv))
c(RMSE = rmse.ridge)
#rmse=0.6533 r2=0.6019
#not better than the rest.larger mse,smaller r2

#LASSO REGRESSION
alpha=1
# Build the model
set.seed(1)
lasso <- train(
  medv ~., data = train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda)
)

# Model coefficients
coef(lasso$finalModel, lasso$bestTune$lambda)
# Make predictions
predictions <- lasso %>% predict(test)
# Model prediction performance

 Rsquare = R2(predictions, test$medv)
 Rsquare
 rmse.lasso<- sqrt(sum(((predictions) - test$medv)^2)/
   length(test$medv))
 c(RMSE = rmse.lasso)
 #     RMSE Rsquare 
# 0.6519 0.6020
#this model still suggests we drop dis like the lm
 #less error than ridge,slight increase in r2
 
 #ELASTIC NET REGRESSION
 # Build the model
 set.seed(1)
 elastic <- train(
   medv ~., data = train, method = "glmnet",
   trControl = trainControl("cv", number = 10),
   tuneLength = 10
 )
 # Model coefficients
 coef(elastic$finalModel, elastic$bestTune$lambda)
 # Make predictions
 predictions <- elastic %>% predict(test)
 # Model prediction performance
 data.frame(
   Rsquare = R2(predictions, test$medv)
 )
 rmse.elastic<- sqrt(sum(((predictions) - test$medv)^2)/
   length(test$medv))
 c(RMSE = rmse.elastic)
 
#    RMSE    Rsquare 
# 0.6519  0.601939
 #same rmse as lasso

 #KNN REGRESSION
 library(caret)
 #training the model
 set.seed(1)
 knn.model= train(
   medv~., data = train, method = "knn",
   trControl = trainControl("cv", number = 10),
   tuneLength = 10
 )
 print(knn.model) #rmse =4.7909,r2=0.7238,final k=7
 # Plot model error RMSE vs different values of k
 plot(knn.model)
 # from the results of the model,Best tuning parameter k that minimize the RMSE is k value of 5
 knn.model$bestTune #k=5         
 # Make predictions on the test data
 predictions <- knn.model %>% predict(test)
 head(predictions)
 # Compute the MSE and r2
 data.frame(
   Rsquare = R2(predictions, test$medv)
 )
 rmse.knn<- sqrt(sum(((predictions) - test$medv)^2)/
   length(test$medv))
 c(RMSE =rmse.knn)
 #RMSE        Rsquare  
 #0.6332   0.6321143
 #smaller rmse,bigger r2 than the preceding model
 
 #DECISION TREES
 #using caret
 set.seed(1)
 control = trainControl(method="cv", number=2)
 tree = train(medv~., data=train, method="rpart", metric="RMSE", trControl=control)
 summary(tree)
 ##MSE=0.2553 at node=5
 
 #predict on test data
 predictions <- tree %>% predict(test)
 # Compute the RMSE and r2
 data.frame(
   Rsquare = R2(predictions, test$medv)
 )
 rmse.tree<- sqrt(sum(((predictions) - test$medv)^2)/
   length(test$medv))
 c(RMSE = rmse.tree) 
 #     RMSE    Rsquare  
 #0.8101   0.4228006
 #worst error so far..
 
 
 library(tree)
 #fit decision tree model on train data
 tree.boston <- tree(medv~.,data = train) #shows the lowest rmse at k= 5
 
 summary(tree.boston)
 
 # Residual mean deviance:0.1892
 #rm,dis,tax,ptratio variables have been used, and our tree has 8 leaves.
 #predict on test data
 predictions <- tree.boston %>% predict(test)
 # Compute the RMSE and r2
 data.frame(
   Rsquare = R2(predictions, test$medv)
 )
 rmse.tree.boston<- sqrt(sum(((predictions) - test$medv)^2)/
                    length(test$medv))
 c(RMSE = rmse.tree.boston) 
 #RMSE           Rsquare
#0.7274337     0.5171437    

 #tree plot
 plot(tree.boston)
 text(tree.boston, pretty = 0)
 #improve model```
 #check to use cv.tree() function to see if crossvalidation will improve the tree- performed worst with cv
 #pruning? cross validation selects the most complex tree i.e. no pruning necessary because the tree hasnt got a lot of branches.
 
 #GRADIENT BOOSTING
 # Fit the model on the training set
 set.seed(1)
xgb.model <- train(
   medv ~., data = train, method = "xgbTree",
   trControl = trainControl("cv", number = 10)
 )
 # Best tuning parameter mtry
 xgb.model$bestTune
 # Make predictions on the test data
 predictions <- xgb.model %>% predict(test)
 
 # Compute the MSE and r2
 rmse.xgb.model<- sqrt(sum(((predictions) - test$medv)^2)/
   length(test$medv))
 c(RMSE = rmse.xgb.model) 
 
 data.frame(
   Rsquare = R2(predictions, test$medv)
 )
#rmse=0.5999,mse=0.3598,r2=0.6655


# Variable important plot
 importance(bag.rf)
 varImpPlot(bag.rf)
 #this easily confirms rm as the most important variable in predicting medv and black as the least important as suggested by other methods in the analysis.

 
 





