#' bagging sample code

#' testing our model
#' -----------------------------------------------------------
#' Pima Indians dataset

# remember to install import from cran.
import::from(bagging.R, bagging, predict.bagging)

library(rpart)
library(mlbench)
library(tidyverse)
library(purrr)
library(ROCR)
data("PimaIndiansDiabetes")

PimaIndiansDiabetes$diabetes %>% table
frac <- 0.7
num_models <- 10

my_bagging <- bagging(diabetes ~ ., PimaIndiansDiabetes, frac, num_models)
my_bagging_pred <- predict(my_bagging, PimaIndiansDiabetes)

# check auc
pred <- prediction(my_bagging_pred$pred, PimaIndiansDiabetes$diabetes)
auc.tmp <- performance(pred, "auc"); 
# AUC for a bagged prediction: 0.897417910447761
cat(paste0("AUC for a bagged prediction: ", as.numeric(auc.tmp@y.values)))

# auc for a random model
# look at `my_bagging_pred`
pred <- prediction(my_bagging_pred$x1, PimaIndiansDiabetes$diabetes)
auc.tmp <- performance(pred, "auc"); 
# AUC for a random model: 0.851843283582089
cat(paste0("AUC for a random model: ", as.numeric(auc.tmp@y.values)))

# test on iris data set

iris_prep <- iris[51:150, ]
iris_prep$Species <- factor(iris_prep$Species)
iris_bagging <- bagging(Species ~ ., iris_prep, 0.8, 5)
iris_bagging_pred <- predict(iris_bagging, iris_prep)

# check auc
pred <- prediction(iris_bagging_pred$pred, iris_prep$Species)
auc.tmp <- performance(pred, "auc"); 
# AUC for a bagged prediction: 0.9836
cat(paste0("AUC for a bagged prediction: ", as.numeric(auc.tmp@y.values)))

# auc for a random model
pred <- prediction(iris_bagging_pred$x1, iris_prep$Species)
auc.tmp <- performance(pred, "auc"); 
# AUC for a random model: 0.93
cat(paste0("AUC for a random model: ", as.numeric(auc.tmp@y.values)))

