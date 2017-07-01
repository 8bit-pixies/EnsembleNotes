#' bagging example

library(rpart)
library(mlbench)
library(tidyverse)
library(purrr)
library(ROCR)

#' simple bagging model (strictly speaking they do CV)
bagging <- function(formula, data, frac, num_models) {
  get_frac <- function(data, response, frac) {
    data %>% 
      group_by_(response) %>% 
      sample_frac(frac) %>% 
      ungroup
  }
  train_obj <- map(1:num_models, ~list(mod=rpart(formula, 
                                                 data=get_frac(data, all.vars(formula)[[1]], frac)), 
                                       id=.x))
  class(train_obj) <- c("bagging", class(train_obj))
  return(train_obj)
}

#' this only works for binary classification for rpart
predict.bagging <- function(bag_obj, data) {
  all_pred <- map(bag_obj, function(x) {
    pred_vals <- as.vector(predict(x$mod, data)[,2]) # this pulls out Pr(Positive)
    dat <- data.frame(pred_vals)
    names(dat) <- paste0("x",x$id)
    return(dat)
  }) %>%
    bind_cols %>%
    mutate(pred = rowMeans(.))
}

