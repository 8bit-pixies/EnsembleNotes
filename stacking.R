#' stacking pure R example

library(tidyverse)
library(rpart)

train <- read_csv("higgs_train_10k.csv")
test <- read_csv("higgs_test_5k.csv")
y <- "response"

rpart_mod <- rpart(response ~., train)
glm_mod <- glm(response ~., train, family="binomial")

train$rpart <- predict(rpart_mod, train)
train$glm <- predict(glm_mod, train)

# stack models together
stack_mod <- glm(response ~ rpart + glm -1, train, family="binomial")
train$stack <- predict(stack_mod, train)

library(ROCR)
glm_performance <- prediction(train$glm, train$response)
cat(paste0("GLM AUC: ", performance(glm_performance, "auc")@y.values))

rpart_performance <- prediction(train$rpart, train$response)
cat(paste0("RPART AUC: ", performance(rpart_performance, "auc")@y.values))

stack_performance <- prediction(train$stack, train$response)
cat(paste0("Stack AUC: ", performance(stack_performance, "auc")@y.values))

