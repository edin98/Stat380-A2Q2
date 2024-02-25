library(ISLR)

#summary(Credit)
head(Credit)
dim(Credit)
str(Credit)


#Setting the categorical variables of the dataset as factors to ensure 
#that R recognizes them as categorical and handles them appropriately during modeling
#Basically creating dummy variables for when modelling
Credit$Student <- as.factor(Credit$Student)
Credit$Ethnicity <- as.factor(Credit$Ethnicity)
Credit$Gender <- as.factor(Credit$Gender)
Credit$Married <- as.factor(Credit$Married)
Credit <- subset(Credit, select = -ID)


#Making a copy to save
Credit1 <- Credit

set.seed(1)

#Setting the variables
library(boot)
dim(Credit)
#We want to split the data in 5 and do 5-fold cross validation to calculate the RMSE







#Part (a): Calculate the 5-fold cross-validation root mean square error (RMSE)
#Creating the model with all predictors
lm_model <- glm(Balance ~ .,data=Credit)
#Cross-validating the model on the entire dataset
cv_err <- cv.glm(data=Credit ,lm_model, K=5)
#The two numbers in the delta vector contain the cross-validation results.
cv_err$delta
print(paste("5-Fold Cross-Validation RMSE of the linear model with all predictors:", sqrt(cv_err$delta[1])))








#Part (b): Run a linear regression of Balance on all other predictors on the full dataset.
lm_model_all <- lm(Balance ~ ., data = Credit)
summary(lm_model_all)
#We can observe that the parameters that are significant at 5% confidence level are the ones the have a p-value 
#less than 5%. This means that it is very likely that the parameter is not equal to zero (has no effect on the model)
#The parameters that are statistically significant are:
#Income, limit, Rating, Cards, Age, and Student.

#Just for fun, I calculated the training RMSE
print(paste("The training RMSE of the linear model with all predictors on full dataset:", sqrt(mean((residuals(lm_model_all))^2))))
#This makes sense, since its evaluating the model on the same dataset it was trained on. Without cross-validation.

#Remember that this is just the MSE on training model, which is not a good indicator of a good model

#Calculate the 5-fold cross-validation RMSE for a linear regression model where Balance
#is regressed only on predictors which were deemed significant in the previous step.

#Statistical Significant Predictors:
independent_vars <- c("Income", "Limit", "Rating", "Cards", "Age", "Student")
dependent_var <- "Balance"
dim(Credit[, c(independent_vars,dependent_var)])

lm_model_2 <- glm(Balance ~ .,data=Credit[, c(independent_vars,dependent_var)])
summary(lm_model_2)

cv_err_2 <- cv.glm(Credit[, c(independent_vars,dependent_var)],lm_model_2, K=5)
cv_err_2$delta
print(paste("5-Fold Cross-Validation RMSE on statistical significant predictors only:", sqrt(cv_err_2$delta[1])))

#We can conclude that the model with only the significant parameters is better, to 
#the model that uses all the parameters. It is less computational heavy too. 








#Part (c):
#Giving too many parameters can lead to over fitting, and trying to find the right combination of parameters 
#to find the best model is computational impossible sometimes. For this reason, we will use
#forward step-wise approach to find the right combination of parameters, and the BIC metric to see the performance of every model.  
library(leaps)
independent_vars <- setdiff(names(Credit), c("Balance")) #I am taking these columns out from independant variables
forward_models <- regsubsets(Balance ~ ., data = Credit, nvmax = length(independent_vars), method = "forward")
summary(forward_models)
summary(forward_models)$bic

best_subset1 <- which.min(summary(forward_models)$bic)
forward_subset_var <- names(coef(forward_models, id = best_subset1))
cat("Number of parameters that minimize BIC:", best_subset1, "\n")
cat("Names of parameters:", forward_subset_var, "\n")
cat("The BIC for best forward model is:", summary(forward_models)$bic[5], "\n")


#this time use an exhaustive search, instead of a forward approach.
exhaustive_models <- regsubsets(Balance ~ ., data = Credit, nvmax = length(independent_vars), method = "exhaustive")
summary(exhaustive_models)
summary(exhaustive_models)$bic

best_subset2 <- which.min(summary(exhaustive_models)$bic)
exhaustive_subset_var <- names(coef(exhaustive_models, id = best_subset2))
cat("Number of parameters that minimize BIC:", best_subset2, "\n")
cat("Names of parameters:", exhaustive_subset_var, "\n")
cat("The BIC for best exhaustive model is:", summary(exhaustive_models)$bic[4], "\n")

#CONCLUSION:  The two methods give different models that have almost the same BIC.
#According to the exhaustive method, the best parameters are Income, Limit, Cards, and Student! 

#Compare the 5-fold cross validation RMSE performance of the final models obtained through the exhaustive search and forward procedure.
#Forward: Income, Limit, Rating, Cards, StudentYES 
lm_model_3 <- glm(Balance ~ Income + Limit + Rating + Cards + Student, data = Credit)
summary(lm_model_3)

cv_err_3 <- cv.glm(Credit,lm_model_3, K=5)
cv_err_3$delta
print(paste("5-Fold Cross-Validation RMSE of forward model selection linear model :", sqrt(cv_err_3$delta[1])))
#The RMSE increased, compared to linear model with significant predictors

#Exhaustive: Income, Limit, Cards, StudentYES 
lm_model_4 <- glm(Balance ~ Income + Limit + Cards + Student, data = Credit)
summary(lm_model_4)

cv_err_4 <- cv.glm(Credit,lm_model_4, K=5)
cv_err_4$delta
print(paste("5-Fold Cross-Validation RMSE of exhaustive model selection linear model :", sqrt(cv_err_4$delta[1])))

cv.glm(as.data.frame(X),lm_model_4, K=5)

#Conclusion: The RMSE is almost the same, but with less parameters. Therefore, the exhaustive search
#parameters that we found are preferred, because there are less. 

#Part (d):
#Restart the kernel or make a copy of Credit dataset
library(glmnet)
set.seed(1)
lambdagrid <- 10^seq(10,-2,length=100)
length(lambdagrid)

#obtain a fully numerical design matrix for predictors

X <- model.matrix(Balance ~ ., data=Credit1)[, -1]
y <- Credit1$Balance
length(y)
dim(X)
head(X)

#Use the function glmnet to calculate the 5-fold RMSE of a lasso regression for all possible values for λ found in lambdagrid.
#If alpha=0 then a ridge regression model is fit, and if alpha=1 then a lasso model is fit.
ridge_mod <- glmnet(X, y, alpha = 0, lambda = lambdagrid, standardize = TRUE)
dim(coef(ridge_mod))

#For lambda_50
ridge_mod$lambda[60]
coef(ridge_mod)[, 60]
sqrt(sum(coef(ridge_mod)[-1, 60]^2))


#Use the function glmnet to calculate the 5-fold RMSE of a ridge regression for all possible values for λ found in lambdagrid

#This is just the predict function to make prediction in our trained model for any lambda, in this case lambda#60
predict(ridge_mod , s = 705.4802, type = "coefficients")[1:12, ]

#So far, I've used the whole data set to make the ridge regression model. I will know do a train-test split on the data.
#So I can do training and cross-validation on data. Then test on the different models, which have different lambdas. 
#to see which one has the lowest RMSE! 

test <- sample(1:nrow(X), nrow(X) / 5)
train <- (-test)

X_train <- X[train , ]
X_test <- X[test, ]

y_train <- y[train]
y_test <- y[test]

dim(X)
dim(X_train)
dim(X_test)
#Did a 80-20 split, since I will need to cross-validate the training set. 

#Now I is the model training on the training data
ridge_mod1 <- glmnet(X_train, y_train, alpha = 0, lambda = lambdagrid)

#Now we need to check which lambda is best, using cross-validation! 
cv_out1 <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 5, lambda = lambdagrid)
plot(cv_out1)
cv_out1


bestlam1 <- cv_out1$lambda.min
bestlam1

#Now that we have the value of the best lambda, according to cross-validation, we can predict
#the y_values using ridge regression with our trained model on trained data. Using the new out-of-sample data (X_train)
ridge_pred1 <- predict(ridge_mod1 , s = bestlam1 , newx = X_test)
#Now we compare the our predicted values associated with the best lambda, and the actual values (y_test), by calculating MSE

print(paste("The best lambda for ridge regression is :", bestlam1))
print(paste("The RMSE on the test data of ridge regression :", sqrt(mean((ridge_pred1 - y_test)^2))))


#According to our model, the best lambda has the value of 2.009233, and its associated 
#RMSE is 102.63











#Part (e):

#Use the function glmnet to calculate the 5-fold RMSE of a lasso regression for all possible values for λ found in lambdagrid.
#If alpha=0 then a ridge regression model is fit, and if alpha=1 then a lasso model is fit.


#Now I is the model training on the training data
lasso_mod2 <- glmnet(X_train, y_train, alpha = 1, lambda = lambdagrid)

#We need to check which lambda is best, using cross-validation! 
cv_out2 <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 5, lambda = lambdagrid)
plot(cv_out2)
cv_out2


bestlam2 <- cv_out2$lambda.min
bestlam2

#Now that we have the value of the best lambda, according to cross-validation, we can predict
#the y_values using lasso regression with our trained model on trained data. Using the new out-of-sample data (X_train)
lasso_pred2 <- predict(lasso_mod2 , s = bestlam2 , newx = X_test)
#Now we compare the our predicted values associated with the best lambda, and the actual values (y_test), by calculating MSE

print(paste("The best lambda for lasso regression is :", bestlam2))
print(paste("The RMSE on the test data of lasso regression :", sqrt(mean((lasso_pred2- y_test)^2))))

#RMSE is 102.8676
out <- glmnet(X, y, alpha = 1, lambda = lambdagrid, standardize = TRUE)
lasso_coef <- predict(out, type = "coefficients", s=bestlam)[1:12,]
lasso_coef









#part (f):
#Out of the first four models (excluding ridge and lasso):

print(paste("5-Fold Cross-Validation RMSE of the linear model with all predictors:", sqrt(cv_err$delta[1])))
print(paste("5-Fold Cross-Validation RMSE on statistical significant predictors only:", sqrt(cv_err_2$delta[1])))
print(paste("5-Fold Cross-Validation RMSE of forward model selection linear model :", sqrt(cv_err_3$delta[1])))
print(paste("5-Fold Cross-Validation RMSE of exhaustive model selection linear model :", sqrt(cv_err_4$delta[1])))

#We can observe that the model with the smallest RMSE is the linear model where coefficients where chosen
#forward method model selection method, so we choose that model. 

#Between Lasso and Ridge regression: 

print(paste("The RMSE on the test data of ridge regression :", sqrt(mean((ridge_pred1 - y_test)^2))))
print(paste("The RMSE on the test data of lasso regression :", sqrt(mean((lasso_pred2- y_test)^2))))

#We can observe that lasso regression had a lesser RMSE, so we choose that model


#Now the way the linear model by forward selection and lasso regression were evaluated are different. 
#Linear regression was evaluated with 5-fold cross-validation on the entire dataset. 
#Lasso regression was evaluated on 20% of the dataset (test dataset)
#So comparing the two methods RMSE is not fair, since it is like they were tested on different datasets. 


#So what we can do is to evaluate the linear model, on how well it will do on the test data

#The linear model by the forward method selection
lm_model_3

#A lambda of value zero is just simply the linear regression (no ridge or lasso)
preds <- predict(lm_model_3, newx = X_test, s = 1)
print(paste("The RMSE on the test data of linear regression :", (mean((preds- y_test)^2))))

