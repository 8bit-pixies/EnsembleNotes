#' boosting

library(rpart)
library(mlbench)
library(tidyverse)
library(purrr)
library(ROCR)

#' simple bagging model (strictly speaking they do CV)
boosting <- function(formula, data, frac, num_models, reweight=1.1) {
  normalise <- function(vec) {
    vec/sum(vec)
  }
  
  index_weight <- normalise(rep(1, nrow(data)))
  mod_list <- list()
  
  for(i in 1:num_models) {
    #' take sample and build base model (Step 1 and 2)
    samp_idx <- sample(seq_len(nrow(data)), nrow(data)*frac, prob=index_weight, replace=TRUE)
    new_model <- rpart(formula, data=data[samp_idx,])
    mod_list[[i]] <- list(mod=new_model, id=i)
    reweight_idx <- unique(samp_idx)
    
    #' update the weights (Step 3)
    new_pred <- predict(new_model, data[reweight_idx, ], type="class")
    
    for(idx in unique(samp_idx)){
      if (predict(new_model, data[idx, ], type="class") == data[idx, all.vars(formula)[[1]]]){
        index_weight[idx] <- index_weight[idx] * (1/reweight)
      } else {
        index_weight[idx] <- index_weight[idx] * reweight
      }
    }
    index_weight <- normalise(index_weight)
  }
  class(mod_list) <- c("boosting", class(mod_list))
  return(mod_list)
}

#' this only works for binary classification for rpart
#' this one is just an average - weights do not change
#' think why did i do it this way?
predict.boosting <- function(bag_obj, data) {
  all_pred <- map(bag_obj, function(x) {
    pred_vals <- as.vector(predict(x$mod, data)[,2]) # this pulls out Pr(Positive)
    dat <- data.frame(pred_vals)
    names(dat) <- paste0("x",x$id)
    return(dat)
  }) %>%
    bind_cols %>%
    mutate(pred = rowMeans(.))
}

