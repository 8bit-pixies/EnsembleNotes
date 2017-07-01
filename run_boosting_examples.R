#' run_boosting_examples.R

import::from(boosting.R, boosting, predict.boosting)

library(rpart)
library(mlbench)
library(tidyverse)
library(purrr)
library(ROCR)
data("PimaIndiansDiabetes")

PimaIndiansDiabetes$diabetes %>% table
frac <- 0.7
num_models <- 10
reweight <- 1.2

my_boost <- boosting(diabetes ~ ., PimaIndiansDiabetes, frac, num_models)
my_boost_pred <- predict(my_boost, PimaIndiansDiabetes)

# check auc
pred <- prediction(my_boost_pred$pred, PimaIndiansDiabetes$diabetes)
auc.tmp <- performance(pred, "auc"); 
# AUC for a boost prediction: 0.917097014925373
cat(paste0("AUC for a boost prediction: ", as.numeric(auc.tmp@y.values)))

# auc for a random model
pred <- prediction(my_boost_pred$x1, PimaIndiansDiabetes$diabetes)
auc.tmp <- performance(pred, "auc"); 
# AUC for a random model: 0.804029850746269
cat(paste0("AUC for a random model: ", as.numeric(auc.tmp@y.values)))


# test on iris data set
iris_prep <- iris[51:150, ]
iris_prep$Species <- factor(iris_prep$Species)
iris_boost <- bagging(Species ~ ., iris_prep, 0.8, 5)
iris_boost_pred <- predict(iris_boost, iris_prep)

# check auc
pred <- prediction(iris_boost_pred$pred, iris_prep$Species)
auc.tmp <- performance(pred, "auc"); 
# AUC for a bagged prediction: 0.9838
cat(paste0("AUC for a bagged prediction: ", as.numeric(auc.tmp@y.values)))

# auc for a random model
pred <- prediction(iris_boost_pred$x1, iris_prep$Species)
auc.tmp <- performance(pred, "auc"); 
# AUC for a random model: 0.93
cat(paste0("AUC for a random model: ", as.numeric(auc.tmp@y.values)))
