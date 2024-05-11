STAT 1361 FINAL PROJECT
================
Philip Kim
4/12/2024

Remove existing variables from environment

``` r
rm(list = ls())
```

Load in necessary libraries

``` r
library(ggplot2)
library(car)
library(leaps)
library(gridExtra)
library(glmnet)
library(class)
library(pls)
library(splines)
library(randomForest)
library(caTools)
library(caret)
library(tree)
library(MASS)
library(cv)
library(dplyr)
library(corrplot)
```

Read in training and test data

``` r
train <- read.csv('train.csv', header = T)
test <- read.csv('test.csv', header = T)
```

Check if there are any missing values in the train/test sets

``` r
cat('Training set has', sum(is.na(train)), 'missing values', '\n')
```

    ## Training set has 0 missing values

``` r
cat('Test set has', sum(is.na(test)), 'missing values')
```

    ## Test set has 0 missing values

Cleaning up Time Signature

``` r
# should range only from 3-7; there are incorrect values in train and test set
train <- subset(train, time_signature != 0)
```

# Exploratory Data Analysis (EDA)

Response Variable (popularity)

``` r
# histogram of popularity - extremely right skewed with large gap in the data
train %>%
  ggplot(aes(x = popularity)) +
  geom_histogram(bins = 20) +
  labs(title = 'Distribution of Popularity', x = "Popularity", y = "Count")
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
# count of zero's found in the popularity column
train %>%
  mutate(zero = ifelse(train$popularity == 0, "zero", "non-zero")) %>%
  ggplot(aes(x = as.factor(zero), fill = as.factor(zero))) +
  geom_bar() +
  scale_fill_brewer(palette = "Pastel2") +
  labs(title = "Count of Zero's vs Non-Zeros in Popularity", x = "Popularity", y = "Count", fill = "Popularity")
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
# proportion of popularity that is 0
cat('Proportion of Popularity is that 0:', sum(train$popularity == 0) / nrow(train))
```

    ## Proportion of Popularity is that 0: 0.4462052

Boxplots of Popularity Across Factor Variables

``` r
# Explicit boxplots
explicit_box <- train %>%
  ggplot(aes(x = as.factor(explicit), y = popularity, fill = as.factor(explicit))) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greens") +
  labs(title = 'Distribution of Popularity Among Explicitity', x = 'Explicit', y = 'Popularity') +
  scale_x_discrete(labels = c("0" = "Not Explicit", "1" = "Explicit")) +
  theme(legend.position = 'none')

# Mode boxplots
mode_box <- train %>%
  ggplot(aes(x = as.factor(mode), y = popularity, fill = as.factor(mode))) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Blues") +
  labs(title = 'Distribution of Popularity Among Mode', x = 'Mode', y = 'Popularity') +
  scale_x_discrete(labels = c("0" = "Minor", "1" = "Major")) +
  theme(legend.position = 'none')


# Time signature boxplots
sig_box <- train %>%
  ggplot(aes(x = as.factor(time_signature), y = popularity, fill = as.factor(time_signature))) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Reds") +
  labs(title = 'Distribution of Popularity Among Time Signatures', x = 'Time Signature', y = 'Popularity') +
  theme(legend.position = 'none')
  
# Key boxplots
key_box <- train %>%
  ggplot(aes(x = as.factor(key), y = popularity, fill = as.factor(key))) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = 'Distribution of Popularity Among Keys', x = 'Key', y = 'Popularity') +
  theme(legend.position = 'none')
  

# boxplots by track_genre - each genre has large gap (most songs in each genre have low popularity, but few have high popularity)
genre_box <- train %>%
  ggplot(aes(x = as.factor(track_genre), y = popularity, fill = track_genre)) +
  scale_fill_brewer(palette = "Pastel1") +  
  geom_boxplot() +
  labs(title = 'Distribution of Popularity Among Track Genres', x = "Track Genre", y = "Popularity") +
  theme(legend.position = 'none')
  
grid.arrange(explicit_box, mode_box, sig_box, key_box, genre_box, ncol = 2)
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

One-hot coding for “explicit” variable

``` r
train$explicit <- ifelse(train$explicit == "TRUE", 1, 0)
test$explicit <- ifelse(test$explicit == "TRUE", 1, 0)
```

Correlation Heatmap

``` r
# the correlation heatmap gives us a good idea on what is correlated with popularity
# we see some multicollinearity occuring in the data - energy is > 4

cor_df <- train[,-c(1:3, 19)]

cor_df$rock <- ifelse(train$track_genre == 'rock', 1, 0)
cor_df$pop <- ifelse(train$track_genre == 'pop', 1, 0)
cor_df$jazz <- ifelse(train$track_genre == 'jazz', 1, 0)

cor_mat <- round(cor(cor_df), 2)

corrplot(cor_mat, method = 'color', type = 'lower', diag = F, tl.cex = 0.5, tl.col = 'black',
         tl.srt = 45, addgrid.col = 'gray')
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

# Multiple Linear Regression

``` r
# train set for the linear regression
lm_df <- train[,-c(1:3, 8)]

# converting necessary variables into factor variables
lm_df$explicit <- as.factor(lm_df$explicit)
lm_df$mode <- as.factor(lm_df$mode)
lm_df$time_signature <- as.factor(lm_df$time_signature)
lm_df$key <- as.factor(lm_df$key)

# run the model
model <- lm(data = lm_df, popularity ~.)
```

Variance Inflation Factor (VIF)

``` r
# finding VIF's for the multiple linear regression
# all are < 4 after removing energy 
vif(model)
```

    ##                      GVIF Df GVIF^(1/(2*Df))
    ## duration_ms      1.203649  1        1.097109
    ## explicit         1.072776  1        1.035749
    ## danceability     1.821553  1        1.349649
    ## key              1.763108 11        1.026111
    ## loudness         2.141723  1        1.463463
    ## mode             1.222297  1        1.105575
    ## speechiness      1.232971  1        1.110392
    ## acousticness     2.545985  1        1.595614
    ## instrumentalness 1.146298  1        1.070653
    ## liveness         1.082009  1        1.040197
    ## valence          1.726150  1        1.313830
    ## tempo            1.260209  1        1.122590
    ## time_signature   1.537135  3        1.074283
    ## track_genre      2.608284  2        1.270834

Cross Validated/Forward Selection Linear Regression Model

``` r
set.seed(123)

# training data for linear regression
lm_df <- train[,-c(1:3, 8)]

# converting necessary variables into factor variables
lm_df$explicit <- as.factor(lm_df$explicit)
lm_df$mode <- as.factor(lm_df$mode)
lm_df$time_signature <- as.factor(lm_df$time_signature)
lm_df$key <- as.factor(lm_df$key)

ctrl <- trainControl(method = 'cv', number = 5, verboseIter = F)
lm_model <- train(popularity ~ ., data = lm_df, method = "lmStepAIC", trControl = ctrl,
                  trace = F)

lm_test_mse <- (lm_model$results$RMSE)^2

summary(lm_model$finalModel)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ duration_ms + explicit1 + danceability + 
    ##     key2 + key10 + mode1 + valence + tempo + time_signature3 + 
    ##     time_signature4 + time_signature5 + track_genrepop, data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -69.594 -18.996  -8.902  22.301  78.327 
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     -1.820e+01  9.822e+00  -1.853  0.06407 .  
    ## duration_ms      6.799e-05  1.481e-05   4.589 4.92e-06 ***
    ## explicit1        7.659e+00  4.350e+00   1.761  0.07857 .  
    ## danceability     2.079e+01  7.546e+00   2.756  0.00595 ** 
    ## key2             1.146e+01  3.477e+00   3.297  0.00101 ** 
    ## key10            1.087e+01  3.515e+00   3.093  0.00203 ** 
    ## mode1           -3.964e+00  1.877e+00  -2.113  0.03484 *  
    ## valence         -1.497e+01  4.647e+00  -3.222  0.00131 ** 
    ## tempo           -5.503e-02  2.843e-02  -1.935  0.05317 .  
    ## time_signature3  2.233e+01  8.504e+00   2.625  0.00877 ** 
    ## time_signature4  2.682e+01  8.169e+00   3.283  0.00106 ** 
    ## time_signature5  2.225e+01  1.126e+01   1.975  0.04846 *  
    ## track_genrepop   2.594e+01  2.019e+00  12.845  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 29.67 on 1186 degrees of freedom
    ## Multiple R-squared:  0.2412, Adjusted R-squared:  0.2335 
    ## F-statistic: 31.41 on 12 and 1186 DF,  p-value: < 2.2e-16

``` r
cat("Linear Regression Test MSE:", lm_test_mse)
```

    ## Linear Regression Test MSE: 905.2293

# K-Nearest Neighbors

One hot coding for all categorical variables

``` r
knn_train <- train
knn_test <- test

# Training one hot coding
for (i in 1:11){
  knn_train[[paste0("key", i)]] <- ifelse(knn_train$key == i, 1, 0)
}

knn_train$pop <- ifelse(knn_train$track_genre == 'pop', 1, 0)
knn_train$rock <- ifelse(knn_train$track_genre == 'rock', 1, 0)
knn_train$jazz <- ifelse(knn_train$track_genre == 'jazz', 1, 0)

knn_train$explicityes <- ifelse(knn_train$explicit == 1, 1, 0)
knn_train$explicitno <- ifelse(knn_train$explicit == 0, 1, 0)

knn_train$major <- ifelse(knn_train$mode == 1, 1, 0)
knn_train$minor <- ifelse(knn_train$mode == 0, 1, 0)

knn_train$ts1 <- ifelse(knn_train$time_signature == 1, 1, 0)
knn_train$ts3 <- ifelse(knn_train$time_signature == 3, 1, 0)
knn_train$ts4 <- ifelse(knn_train$time_signature == 4, 1, 0)
knn_train$ts5 <- ifelse(knn_train$time_signature == 5, 1, 0)


# Test set one hot coding
for (i in 1:11){
  knn_test[[paste0("key", i)]] <- ifelse(knn_test$key == i, 1, 0)
}

knn_test$pop <- ifelse(knn_test$track_genre == 'pop', 1, 0)
knn_test$rock <- ifelse(knn_test$track_genre == 'rock', 1, 0)
knn_test$jazz <- ifelse(knn_test$track_genre == 'jazz', 1, 0)

knn_test$explicityes <- ifelse(knn_test$explicit == 1, 1, 0)
knn_test$explicitno <- ifelse(knn_test$explicit == 0, 1, 0)

knn_test$major <- ifelse(knn_test$mode == 1, 1, 0)
knn_test$minor <- ifelse(knn_test$mode == 0, 1, 0)

knn_test$ts1 <- ifelse(knn_test$time_signature == 1, 1, 0)
knn_test$ts3 <- ifelse(knn_test$time_signature == 3, 1, 0)
knn_test$ts4 <- ifelse(knn_test$time_signature == 4, 1, 0)
knn_test$ts5 <- ifelse(knn_test$time_signature == 5, 1, 0)
```

Running the KNN algorithm

``` r
# variables we want to remove from the train and test sets (either irrelevant or redundant due to dummies)
remove_var <- c('id', 'album_name', 'track_name', 'explicit','key', 'mode', 'time_signature', 'track_genre')


# create x and y train, x test data sets for the algorithm
x_train <- knn_train[, !names(knn_train) %in% c(remove_var, 'popularity')]
y_train <- knn_train %>%
  select(popularity)
x_test <- knn_test[, !names(knn_test) %in% remove_var]


# for reproducibility
set.seed(123)


# cross validation to find optimal k
trControl <- trainControl(method = 'cv', number = 5)
knn_cv <- train(x = x_train, y = y_train$popularity, method = 'knn', tuneGrid = expand.grid(k = 1:100),
                 trControl = trControl, preProcess = c("center", "scale"))

knn_results <- knn_cv$results
plot(knn_cv)
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
# optimal k
best_k <- which.min(knn_results$RMSE)

# running knn with the optimal k value found via CV
final_knn <- knnreg(x_train, y_train$popularity, k = best_k)

# predicting test data with the final knn model
knn_test$predictions <- predict(final_knn, newdata = x_test)


# kNN MSE from Cross Validation
min_mse <- knn_results[which.min(knn_results$RMSE),]
knn_mse_value <- (min_mse$RMSE)^2

cat('kNN Cross Validated Minimum MSE:', knn_mse_value, "with k =", best_k, "neighbors")
```

    ## kNN Cross Validated Minimum MSE: 947.7544 with k = 50 neighbors

# PCR

``` r
set.seed(123)

# running PCR model with cross validation
pcr_model = pcr(popularity ~ ., data = train[,-c(1:3)], scale = TRUE, validation = "CV")

# plot that shows MSEP against number of PC
validationplot(pcr_model, val.type = "MSEP")
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
# define the testing set for PCR
pcr_test <- test[,-c(1:2)]
pcr_test$explicit <- ifelse(pcr_test$explicit == "TRUE", 1, 0)

# using the PCR model to predict values on the PCR test set
pcr_pred <- predict(pcr_model, pcr_test, ncomp=5)

pcr_test$predictions <- predict(pcr_model, pcr_test, ncomp=5)



# Cross validated MSE from PCR Training
msep_pcr <- MSEP(pcr_model)

paste( "Minimum MSE of",  
       msep_pcr$val[1,1, ][6][1], 
       "was produced with 5 components")
```

    ## [1] "Minimum MSE of 921.854102744666 was produced with 5 components"

``` r
min_pcr_mse <- msep_pcr$val[1,1, ][6][1]
pcr_mse_value <- as.data.frame(min_pcr_mse)$min_pcr_mse
```

# Random Forest

``` r
set.seed(123)

# defining the train and test sets for random forest
rf_train <- train
rf_test <- test

# converting variables into factor variables for model consistency
rf_train$track_genre <- as.factor(rf_train$track_genre)
rf_test$track_genre <- as.factor(rf_test$track_genre)

rf_train$time_signature <- as.factor(rf_train$time_signature)
rf_test$time_signature <- as.factor(rf_test$time_signature)

rf_train$key <- as.factor(rf_train$key)
rf_test$key <- as.factor(rf_test$key)

# creating and refining the x train, y train, and testing set
remove_var_rf <- c('id', 'album_name', 'track_name')

x_rf_train <- rf_train[, !names(rf_train) %in% c(remove_var_rf, 'popularity')]
y_rf_train <- rf_train %>%
  select(popularity)

rf_train <- rf_train[, !names(rf_train) %in% remove_var_rf]
rf_test <- rf_test[, !names(rf_test) %in% remove_var_rf]


# cross validation so we can find the optimal mtry
trControl_rf <- trainControl(method = 'cv', number = 5)

mtry_grid <- expand.grid(mtry = seq(1:length(rf_train)-1)) # controls number of mtry to consider for the CV
rf_model <- train(x = x_rf_train, y = y_rf_train$popularity, method = 'rf', 
                  tuneGrid = mtry_grid, trControl = trControl_rf)
                                                                                                                      rf_results <- rf_model$results

# optimal mtry
best_mtry <- which.min(rf_results$RMSE)

# fit the random forest model with the optimal mtry
rf.fit <- randomForest(popularity ~., data = rf_train, mtry = best_mtry, ntree = 100, importance = TRUE)


rf_test$predictions <- predict(rf.fit, rf_test)


# variable importance
varImpPlot(rf.fit)
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
varImp(rf.fit) %>%
  arrange(desc(Overall))
```

    ##                    Overall
    ## track_genre      31.228382
    ## key              19.880876
    ## duration_ms      16.306332
    ## danceability     15.768479
    ## valence          13.407772
    ## energy           12.831855
    ## speechiness      12.826184
    ## instrumentalness 12.686551
    ## acousticness     12.271455
    ## loudness         11.531028
    ## tempo            11.419211
    ## liveness         10.380351
    ## mode              4.590040
    ## time_signature    2.928270
    ## explicit          1.267356

``` r
# example of a decision tree 
tree <- tree(popularity ~., data = rf_train)
plot(tree)
text(tree, pretty = 0)
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-16-2.png)<!-- -->

``` r
# minimum MSE from cross validated random forest 
min_mse_rf <- rf_results[which.min(rf_results$RMSE),]
rf_mse_value <- (min_mse_rf$RMSE)^2
cat('Random Forest Cross Validated Minimum MSE:', rf_mse_value, '\n')
```

    ## Random Forest Cross Validated Minimum MSE: 685.7244

``` r
cat('Random Forest Cross Validated Minimum MAE:', min_mse_rf$MAE)
```

    ## Random Forest Cross Validated Minimum MAE: 20.04468

# Boosting

``` r
set.seed(123)

# creating copies of the train and test set just for boosting
boost_train <- train
boost_test <- test

# converting variables into factor variables for model consistency
boost_train$track_genre <- as.factor(boost_train$track_genre)
boost_test$track_genre <- as.factor(boost_test$track_genre)

boost_train$time_signature <- as.factor(boost_train$time_signature)
boost_test$time_signature <- as.factor(boost_test$time_signature)

boost_train$key <- as.factor(boost_train$key)
boost_test$key <- as.factor(boost_test$key)


# creating and refining the x train, y train, and testing set
remove_var_boost <- c('id', 'album_name', 'track_name')

x_boost_train <- boost_train[, !names(boost_train) %in% c(remove_var_boost, 'popularity')]
y_boost_train <- boost_train %>%
  select(popularity)
boost_test <- boost_test[, !names(boost_test) %in% remove_var_boost]

# cross validation for boosting (specifically for number of trees, lambda (learning rate), and depth)
trControl_boost <- trainControl(method = 'cv', number = 5)
boost_grid <- expand.grid(n.trees = seq(100, 1000, by = 50),
                          shrinkage = c(0.01, 0.05, 0.1),
                          interaction.depth = seq(1,15), 
                          n.minobsinnode = 10)

boost_model <- train(x = x_boost_train, y = y_boost_train$popularity, method = 'gbm', tuneGrid = boost_grid, 
                     trControl = trControl_boost, verbose = FALSE)

# results from the cross validation
boost_results <- boost_model$results

# characteristics of the best model
best_tune <- boost_model$bestTune
best_tune
```

    ##     n.trees interaction.depth shrinkage n.minobsinnode
    ## 266    1000                14      0.01             10

``` r
final_boost_model <- boost_model$finalModel

# predicting on the test data with the best final model
boost_predictions <- predict(final_boost_model, newdata = boost_test)
```

    ## Using 1000 trees...

``` r
boost_test$predictions <- predict(final_boost_model, newdata = boost_test)
```

    ## Using 1000 trees...

``` r
# variable importance 
var_importance <- summary(boost_model)
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
var_importance
```

    ##                               var    rel.inf
    ## key                           key 18.1593119
    ## track_genre           track_genre 12.3171103
    ## acousticness         acousticness  8.9033192
    ## duration_ms           duration_ms  7.6954120
    ## danceability         danceability  7.3581755
    ## valence                   valence  7.3337527
    ## speechiness           speechiness  6.7773671
    ## energy                     energy  6.7602169
    ## loudness                 loudness  6.3248755
    ## liveness                 liveness  5.9513758
    ## tempo                       tempo  5.6176121
    ## instrumentalness instrumentalness  5.5222363
    ## mode                         mode  0.6034901
    ## explicit                 explicit  0.5284203
    ## time_signature     time_signature  0.1473244

``` r
# example tree from training (boosting)
tree_boost <- tree(popularity ~., data = boost_train)
plot(tree_boost)
text(tree_boost, pretty = 0)
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-17-2.png)<!-- -->

``` r
# minimum MSE from cross validated random forest 
min_mse_boost <- boost_results[which.min(boost_results$RMSE),]
mse_boost_value <- (min_mse_boost$RMSE)^2
cat('Boosting Cross Validated Minimum MSE:', mse_boost_value)
```

    ## Boosting Cross Validated Minimum MSE: 728.696

# Test MSE Comparisions

All Test MSEs Comparision

``` r
all_errors <- c(lm_test_mse, pcr_mse_value, knn_mse_value, rf_mse_value, mse_boost_value)
names(all_errors) <- c("LR", "PCR", "kNN", "RF", "Boost")
barplot(all_errors, col = 'light blue', main = "Test MSE's from Modeling", xlab = 'Model', ylab = "Test MSE")
```

![](STAT-1361-FINAL-PROJECT_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

Creating the Final Prediction CSV Using Random Forest

``` r
predictions_csv <- data.frame(matrix(nrow = 500, ncol = 2))
predictions_csv[1] <- test[1]
predictions_csv[2] <- round(predict(rf.fit, rf_test),0)
options(scipen = 999)
names(predictions_csv) <- c("id", "popularity")

write.csv(predictions_csv, "testing_predictions_KIM_PHILIP_PJK55.csv", row.names = FALSE)
```
