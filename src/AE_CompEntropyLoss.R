# Train the compressor separatly, with cross-entropy loss function
library(keras)

# Parameters
original_dim <- 784L
compressed_dim <- 256L
batch_size <- 1000L
epochs <- 40L
lambda <- 0.5

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

# Loss function
entr <- c(1:256)
loss_entropy <- function(y_true, y_pred) {
  K <- backend()
  y_pred <- K$eval(y_pred)

}

# Compile the compressor separatly
cae %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.00001),
  loss = loss_entropy
)
