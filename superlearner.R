#' add some examples related to latest version of h2o and the superlearner approach
#' https://h2o-release.s3.amazonaws.com/h2o/rel-tverberg/2/docs-website/h2o-docs/data-science/stacked-ensembles.html


library(h2o)

h2o.init()

# Import a sample binary outcome train/test set into H2O
train <- h2o.importFile("higgs_train_10k.csv")
test <- h2o.importFile("higgs_test_5k.csv")

# Identify predictors and response
y <- "response"
x <- setdiff(names(train), y)

# For binary classification, response should be a factor
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5



# There are a few ways to assemble a list of models to stack toegether:
# 1. Train individual models and put them in a list
# 2. Train a grid of models
# 3. Train several grids of models
# Note: All base models must have the same cross-validation folds and
# the cross-validated predicted values must be kept.



# 1. Generate a 2-model ensemble (GBM + RF)

# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train,
                  distribution = "bernoulli",
                  ntrees = 10,
                  max_depth = 3,
                  min_rows = 2,
                  learn_rate = 0.2,
                  nfolds = nfolds,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)

# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train,
                          ntrees = 50,
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)

# Train & Cross-validate a glm
my_glm <- h2o.glm(x = x, 
                  y = y, 
                  training_frame = train,
                  nfolds = nfolds,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  family="binomial",
                  seed = 1)

# Train a stacked ensemble using the GBM and RF above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                model_id = "my_ensemble_binomial",
                                base_models = list(my_gbm@model_id, my_rf@model_id, my_glm@model_id))

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

# Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = test)
perf_rf_test <- h2o.performance(my_rf, newdata = test)
perf_glm_test <- h2o.performance(my_glm, newdata = test)
baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), h2o.auc(perf_rf_test), h2o.auc(perf_glm_test))
ensemble_auc_test <- h2o.auc(perf)
# Best Base-learner Test AUC:  0.769800386918767
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
# Ensemble Test AUC:  0.777822073675447
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = test)

