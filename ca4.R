install.packages("randomForest")
install.packages("class")
install.packages("caTools")
install.packages("tidyverse")
install.packages("caret")
install.packages("mlbench")
install.packages("rpart")

library(caTools)
library(randomForest)
library(tidyverse)
library(carat)
library(dplyr)
library(ggplot2)
library(lattice)
library(e1071)
library(class)
library(mlbench)
library(rpart)



laptops

laptops <- na.omit(laptops)

if(any(is.na(laptops))){
  stop("Our Dataset contains a null value!!!")
}


set.seed(123)
split <- sample.split(laptops$Price, SplitRatio = 0.80)
train_data <- subset(laptops, split == TRUE)
test_data <- subset(laptops, split == FALSE)

train_data
test_data


if(any(is.na(train_data)) || any(is.na(test_data))){
  stop("Training and Test set contains a null values")
}


#linear regression model
model <- lm(Price ~ ., data = train_data)
predictions <- predict(model, newdata = test_data)
mse <- mean((test_data$Price - predictions)^2)
r_squared <- 1 - mse / var(test_data$Price)
cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r_squared, "\n")


#Random Forest model
model <- randomForest(Price ~ ., data = train_data, ntree = 100)
predictions <- predict(model, newdata = test_data)
mse <- mean((test_data$Price - predictions)^2)
r_squared <- 1 - mse / var(test_data$Price)
cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r_squared, "\n")


#decision tree model
model <- rpart(Price ~ ., data = train_data, method = "anova")
rfpredictions <- predict(model, newdata = test_data)
mse <- mean((test_data$Price - predictions)^2)
r_squared <- 1 - mse / var(test_data$Price)
cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r_squared, "\n")

# SVM regression model
model <- svm(Price ~ ., data = train_data, kernel = "radial")
predictions_svm <- predict(model, newdata = test_data)
mse <- mean((test_data$Price - predictions)^2)
r_squared <- 1 - mse / var(test_data$Price)
cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r_squared, "\n")

# scatter plot
SVM_plot_data <- data.frame(Actual = test_data$Price, Predicted = rfpredictions)
SVM_plot_data$Correct <- SVM_plot_data$Actual == SVM_plot_data$Predicted
# Plot the actual outcomes against the predicted outcomes, using the 'Correct' column to color the points
plot(SVM_plot_data$Actual, SVM_plot_data$Predicted, col = 
       ifelse(SVM_plot_data$Correct, "green", "purple"), pch = 16,
     main = "SVM", xlab = "Actual", ylab = "Predicted")
abline(a=0, b=1, col = "red")

# Jitter Plot for SVM Regression
jitter_plot_data <- data.frame(Actual = test_data$Price, Predicted = predictions_svm)
jitter_plot_data$Correct <- jitter_plot_data$Actual == jitter_plot_data$Predicted

library(ggplot2)

ggplot(jitter_plot_data, aes(x = Actual, y = Predicted, color = Correct)) +
  geom_point(position = position_jitter(width = 0.3, height = 0.3), size = 3) +
  labs(title = "Jitter Plot for SVM Regression",
       x = "Actual Price",
       y = "Predicted Price",
       color = "Correct Prediction") +
  scale_color_manual(values = c("green", "purple")) +
  theme_minimal()

# Convert 'TypeOfLaptop' to a factor
train_data$TypeOfLaptop <- as.factor(train_data$TypeOfLaptop)
test_data$TypeOfLaptop <- as.factor(test_data$TypeOfLaptop)

# Naive Bayes model
nb_model <- naiveBayes(TypeOfLaptop ~ ., data = train_data)

# Predictions on the test set
nb_predictions <- predict(nb_model, newdata = test_data)

# Confusion matrix
confusion_matrix <- table(Actual = test_data$TypeOfLaptop, Predicted = nb_predictions)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")