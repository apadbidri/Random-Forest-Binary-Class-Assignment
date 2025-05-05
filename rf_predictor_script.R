# Load required libraries
library(readr)
library(VIM)
library(randomForest)
library(caret)
library(pROC)

# === Load data ===
train <- read.csv("data/assign3_train.csv")
test <- read.csv("data/assign3_test.csv")

# === Preprocessing ===

# Convert blank strings to NA
train[train == ""] <- NA
test[test == ""] <- NA

# Convert relevant columns to factors
cols_to_factor <- c("x4", "x5", "x16", "x18", "x20")
train[cols_to_factor] <- lapply(train[cols_to_factor], as.factor)
test[cols_to_factor] <- lapply(test[cols_to_factor], as.factor)

# Split train into numeric and factor
train_num <- train[, sapply(train, is.numeric)]
train_fact <- train[, sapply(train, is.factor)]

test_num <- test[, sapply(test, is.numeric)]
test_fact <- test[, sapply(test, is.factor)]

# kNN imputation for numeric columns only
train_num_imp <- kNN(train_num, k = 5, imp_var = FALSE)
test_num_imp <- kNN(test_num, k = 5, imp_var = FALSE)

# Mode imputation for factors
mode_impute <- function(df) {
  for (col in names(df)) {
    if (is.factor(df[[col]])) {
      mode_val <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
      df[[col]][is.na(df[[col]])] <- mode_val
      df[[col]] <- droplevels(df[[col]])
    }
  }
  return(df)
}

train_fact_imp <- mode_impute(train_fact)
test_fact_imp <- mode_impute(test_fact)

# Recombine numeric and factor columns
train_imputed <- cbind(train_num_imp, train_fact_imp)
test_imputed <- cbind(test_num_imp, test_fact_imp)

# === Prepare data for modeling ===

# Drop any leftover logical columns
train_clean <- train_imputed[, !sapply(train_imputed, is.logical)]
test_clean <- test_imputed[, !sapply(test_imputed, is.logical)]

# Ensure target is factor with correct labels
train_clean$y <- factor(train_clean$y, levels = c(0, 1), labels = c("No", "Yes"))

# Align factor levels in test set with train set
for (col in names(train_clean)) {
  if (is.factor(train_clean[[col]]) && col %in% names(test_clean)) {
    test_clean[[col]] <- factor(test_clean[[col]], levels = levels(train_clean[[col]]))
  }
}

# === Split train data for internal validation ===
set.seed(123)
train_index <- createDataPartition(train_clean$y, p = 0.8, list = FALSE)
train_set <- train_clean[train_index, ]
test_set <- train_clean[-train_index, ]

# === Model training with cross-validation ===
set.seed(123)
train_control <- trainControl(method = "cv",
                              number = 5,
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              sampling = "down")  


# Tune mtry
tune_grid <- expand.grid(mtry = c(5, 7, 9))

cv_model <- train(y ~ ., data = train_set, method = "rf",
                  trControl = train_control,
                  metric = "ROC",
                  tuneGrid = tune_grid,
                  ntree = 100)

print(cv_model)

# === Model evaluation ===
best_model <- cv_model$finalModel
test_predictions_prob <- predict(cv_model, test_set, type = "prob")[, "Yes"]
test_predictions <- predict(cv_model, test_set)

conf_matrix <- confusionMatrix(test_predictions, test_set$y)
roc_curve <- roc(test_set$y, test_predictions_prob, levels = c("No", "Yes"), direction = "<")

cat("Accuracy:", conf_matrix$overall['Accuracy'], "\n")
cat("Precision:", conf_matrix$byClass['Pos Pred Value'], "\n")
cat("Recall:", conf_matrix$byClass['Sensitivity'], "\n")
cat("F1-score:", 2 * (conf_matrix$byClass['Pos Pred Value'] * conf_matrix$byClass['Sensitivity']) / 
      (conf_matrix$byClass['Pos Pred Value'] + conf_matrix$byClass['Sensitivity']), "\n")

plot(roc_curve, main = "ROC Curve", col = "blue")

# === Predict on final test set ===
test_final <- test_clean[, names(train_clean)[names(train_clean) != "y"]]

final_predictions <- predict(cv_model, test_final)
final_predictions_numeric <- as.numeric(final_predictions) - 1  # Yes -> 1, No -> 0
predictions_df <- data.frame(y = final_predictions_numeric)
head(predictions_df)


# === Export predictions ===
# write.csv(predictions_df, "2834723.csv", row.names = FALSE)  # Replace with your candidate number!
