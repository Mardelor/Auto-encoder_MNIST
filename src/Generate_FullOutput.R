# Generate Data for Classification
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# AE
x_train <- x_train/255
cae <- load_model_hdf5('CompAE_0.001')
y_train <- predict_on_batch(cae, x_train)

# Apply the empirical distribution function
for (i in c(1:256)) {
  cdf <- ecdf(y_train[,i])
  y_train[,i] <- cdf(y_train[,i])
}

# Checking the distribution
y_df <- data.frame(y_train)
for (i in c(1:256)) {
  plot(ggplot(melt(y_df[,i]), aes(x = value)) + geom_density(alpha = 0.5))
}