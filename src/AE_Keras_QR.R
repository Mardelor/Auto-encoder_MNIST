library(keras)

# Parameters
original_dim <- 784L
compressed_dim <- 256L
batch_size <- 100L
epochs <- 60L

# Preparing data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255
x_test <- x_test / 255

# Model definition
x <- layer_input(shape = c(original_dim))
compress <- layer_dense(x, units = compressed_dim, activation = 'relu', input_shape = c(original_dim))
decompress <- layer_dense(compress, units = original_dim, activation = 'relu')

# Full autoencoder
fae <- keras_model(x, decompress)

# Compressor
cae <- keras_model(x, compress)

# Trained with a variant of sgd : rmsprop
fae %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = 'mse'
)

# Train it
history <- fae %>% fit(
  x_train, x_train,
  epochs = epochs,
  batch_size = batch_size
)

# Repeat with orthonormalization
for (i in c(1,1:4)) {
  
  # Plot loss function
  plot(history)
  
  # Check orthogonality of weight's columns
  weights <- as.matrix((get_weights(cae))[[1]])
  prod <- t(weights) %*% weights
  print(prod[1:5, 1:5])
  
  # Orthonormalization Gram Schmidt
  weights <- get_weights(cae)[[1]]
  weights <- qr.Q(qr(weights))
  set_weights(cae, list(weights, get_weights(cae)[[2]]))
  
  # Train it
  history <- fae %>% fit(
    x_train, x_train,
    epochs = 10,
    batch_size = batch_size
  )
}

# Saving model
save_model_hdf5(fae, 'FullAE_0.001_GramSchmidt')
save_model_hdf5(cae, 'CompAE_0.001_GramSchmidt')

# Check orthogonality of weight's columns
weights <- as.matrix((get_weights(cae))[[1]])
prod <- t(weights) %*% weights
print(prod[1:5, 1:5])

# See distribution
x_sample1000 <- x_train[1:2000,]
y_sample1000 <- predict_on_batch(cae, x_sample1000)
for (i in c(1:10)) {
  hist(y_sample1000[,i], main = "Histogram of activation values")
}

# helper function for visualization
show_digit = function(arr784, col = gray(12:1 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}

# Check Datas
for (i in c(1:20)) {
  show_digit(x_test[i,])
  show_digit(predict_on_batch(fae, t(x_test[i,])))
}