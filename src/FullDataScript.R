# Multiples number of outputs neurons for the Auto encoder (from 32 to 432)
# For each number, compute the loss and the entropy
library(keras)
library(ggplot2)
library(reshape2)
library(entropy)
library(dplyr)

# Output
out <- c()
stories <- list()

# Parameters
original_dim <- 784L
batch_size <- 50L
epochs <- 20L
periods <- 5L

# Preparing data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train_r <- x_train / 255
x_test_r <- x_test / 255

# For all compressed dimension
for (i in c(7:12)) {
  compressed_dim <- 32 + (i-1)*16
  out <- c(out, compressed_dim)

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
  
  # Full autoencoder
  fae <- keras_model(x, decompress(compress))
  
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
      mean[i] <- mean(y_train[,i]) - 128
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
  
  # Compute Loss function
  out <- c(out, evaluate(fae, x_train, x_train))
  
  # Compute entropy
  y_train <- predict_on_batch(cae, x_train)
  y_train <- apply(y_train, 2, round)
  entropy <- c(1:compressed_dim)
  for (i in c(1:compressed_dim)) {
    e <- entropy(table(y_train[,i]), unit = 'log2')
    if (is.na(e)) {
      entropy[i] <- 0
    } else {
      entropy[i] <- e
    }
  }
  out <- c(out, mean(entropy))
  out <- c(out, sd(entropy))
  out <- c(out, sum(entropy)/8)
}

results <- data.frame(matrix(out, byrow = TRUE, nrow = 26))
names(results) <- c('N', 'Loss', 'Entropy Mean', 'Entropy Standart Deviation', 'Total Entropy(Byte)')
ggplot(melt(results[,1:2], id.vars = 'N'), aes(x = N, y = value)) + geom_line()
ggplot(melt(results[,c(TRUE, FALSE, TRUE, FALSE, FALSE)], id.vars = 'N'), aes(x = N, y = value)) + geom_line()
ggplot(melt(results[,c(TRUE, FALSE, FALSE, TRUE, FALSE)], id.vars = 'N'), aes(x = N, y = value)) + geom_line()
ggplot(melt(results[,c(TRUE, FALSE, FALSE, FALSE, TRUE)], id.vars = 'N'), aes(x = N, y = value)) + geom_line()
