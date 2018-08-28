# Custom Loss function : 8 bits entropy
library(keras)

# Parameters
original_dim <- 784L
compressed_dim <- 256L
batch_size <- 100L
epochs <- 50L

# Preparing data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255
x_test <- x_test / 255

# PCA
x_pca <- prcomp(x_rescale, center = TRUE, scale = FALSE)
x_pca$x <- ((x_pca$x + abs(min(x_pca$x)))/max(x_pca$x))

# Model definition
x <- layer_input(shape = c(original_dim))
compress <- layer_dense(x, units = compressed_dim, activation = 'relu', input_shape = c(original_dim))
decompress <- layer_dense(compress, units = original_dim, activation = 'relu')

# Full autoencoder
fae <- keras_model(x, decompress)

# Compressor
cae <- keras_model(x, compress)

# Loss function
loss_8bit_entropy(y_true, y_pred) {
  
}

# Trained with a variant of sgd : rmsprop
fae %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = 'mse'
)
