# Model to encode MNIST Data base with 8 bit entropy and uniform distribution
library(keras)
library(ggplot2)
library(reshape2)
library(entropy)
library(dplyr)

# Parameters
original_dim <- 784L
compressed_dim <- 256L
batch_size <- 50L
epochs <- 20L
periods <- 5L

# Preparing data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# -----------------------------------------------------------------------------------------------------------------
# Initialization of the ANN

# Initialize m and activation functions
mean <- rep(0, compressed_dim)

# Custom activation function
activation_generator <- function(index) {
  function(x) {
    activation_softplus(x-mean[index] + 128)
  }
}

# Model definition
x <- layer_input(shape = c(original_dim), name = 'Input')
comp <- list()
for (i in c(1:compressed_dim)) {
  # Layer for output neuron i
  comp[[i]] <- layer_dense(x, units = 1, 
                           activation = activation_generator(i), 
                           input_shape = original_dim, name = sprintf('Output_neuron_%d', i))
}
compress <- layer_concatenate(as.vector(comp), name = 'Compression_Layer')
decompress <- layer_dense(units = original_dim, activation = activation_softplus, name = 'Decompression_Layer')
y <- layer_input(shape = c(compressed_dim), name = 'Intermediate_Output')

# Compile
cae <- keras_model(x, compress)
dae <- keras_model(y, decompress(y))
fae <- keras_model(x, decompress(compress))
fae %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = 'mse'
)

# Train it
hist <- fae %>% fit(
  x_train, x_train,
  epochs = epochs,
  validation_data = list(x_test, x_test),
  batch_size = batch_size
)

# Save weights
save_model_weights_hdf5(fae, filepath = 'weights')

# ---------------------------------------------------------------------------------------------------------------
# Trainings changing activation functions

for (j in c(1:periods)) {
  # Generate output data
  y_train <- predict_on_batch(cae, x_train)
  y_train <- apply(y_train, 2, round)

  # Compute mean
  for (i in c(1:compressed_dim)) {
    mean[i] <- mean(y_train[,i]) -128
  }
  
  # Building new model
  for (i in c(1:compressed_dim)) {
    
    # Layer for output neuron i
    comp[[i]] <- layer_dense(x, units = 1, 
                             activation = activation_generator(i), 
                             input_shape = original_dim, name = sprintf('Output_neuron_%d', i))
  }
  compress <- layer_concatenate(as.vector(comp), name = 'Compression_Layer')
  decompress <- layer_dense(compress, units = original_dim, activation = activation_softplus, name = 'Decompression_Layer')
  
  # Full autoencoder
  fae <- keras_model(x, decompress)
  
  # Compressor
  cae <- keras_model(x, compress)
  
  # Load weights
  load_model_weights_hdf5(fae, filepath = 'weights')
  
  # Compile with a variant of sgd : rmsprop
  fae %>% compile(
    optimizer = optimizer_rmsprop(lr = 0.001),
    loss = 'mse'
  )
  
  # Train it
  hist <- fae %>% fit(
    x_train, x_train,
    epochs = 5,
    validation_data = list(x_test, x_test),
    batch_size = batch_size
  )
  plot(hist)
  
  # Save weights
  save_model_weights_hdf5(fae, filepath = 'weights')
}

# -----------------------------------------------------------------------------------------------------------------
# Checking results

# Views on test data
y_train <- predict_on_batch(cae, x_train)
y_train <- apply(y_train, 2, round)
y_train_df <- data.frame(y_train)

# Distribution
for (i in c(0:15)) {
  plot(ggplot(melt(y_train_df[,(i*16 + 1):((i+1)*16)]), aes(x = value, fill = variable)) + geom_density(alpha = 0.2))
}

# Entropy
entropyAE <- c(1:compressed_dim)
for (i in c(1:compressed_dim)) {
  e <- entropy(table(y_train[,i]), unit = 'log2')
  if (is.na(e)) {
    entropyAE[i] <- 0
  } else {
    entropyAE[i] <- e
  }
}
plot(entropyAE)
print(mean(entropyAE))
print(sd(entropyAE))

# Apply the empirical distribution function
y <- y_train
for (i in c(1:compressed_dim)) {
  cdf <- ecdf(y_train[,i])
  y[,i] <- (2^(entropyAE[i] - 8))*cdf(y_train[,i])
}

# Checking the distribution
y_df <- data.frame(y)
for (i in c(0:15)) {
  plot(ggplot(melt(y_df[,(i*16 + 1):((i+1)*16)]), aes(x = value, fill = variable)) + geom_density(alpha = 0.2))
}
