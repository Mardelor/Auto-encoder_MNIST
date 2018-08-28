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
firstEpochs <- 50L
epochs <- 6L
periods <- 5L
sigmaRange <- 6L

# Preparing data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# -----------------------------------------------------------------------------------------------------------------
# Initialization of the ANN

# Initialize a, m and activation functions
alpha <- rep(256, compressed_dim)
mean <- rep(0, compressed_dim)

# Custom activation function
activation_generator <- function(index) {
  function(x) {
    256*(x-mean[index])/alpha[index]
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
decompress <- layer_dense(compress, units = original_dim, activation = activation_softplus, name = 'Decompression_Layer')

# Compile
cae <- keras_model(x, compress)
fae <- keras_model(x, decompress)
fae %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = 'mse'
)

# Train it
hist <- fae %>% fit(
  x_train, x_train,
  epochs = firstEpochs,
  validation_data = list(x_test, x_test),
  batch_size = batch_size
)

# Add to the stories
plot(hist)

# Evaluation over tests samples
evaluations <- vector(mode = 'numeric', periods + 1)
evaluations[1] <- evaluate(fae, x_test, x_test)

# Vector which describe for each neuron if the entropy 8 bit is reached or not,
# i.e. if we need to train the corresponding layer
trainability <- !logical(compressed_dim)
trainable <- vector(mode = 'integer', periods + 1)
trainable[1] <- sum(trainability)

# Save weights
save_model_weights_hdf5(fae, filepath = 'weights')

# ---------------------------------------------------------------------------------------------------------------
# Trainings changing activation functions

for (j in c(1:periods)) {
  # Generate output data
  y_train <- predict_on_batch(cae, x_train)
  y_train <- apply(y_train, 2, round)
  
  # Compute entropy for each neuron in order to determine if a neuron needs to be trained or not
  entropyAE <- c(1:compressed_dim)
  for (i in c(1:compressed_dim)) {
    e <- entropy(table(y_train[,i]), unit = 'log2')
    if (is.na(e)) {
      entropyAE[i] <- 0
    } else {
      entropyAE[i] <- e
    }
  }
  
  # Update Trainability vector
  trainability <- entropyAE > 8
  print(sprintf('Period %d - Trainable: %d', j, sum(trainability)))
  append(trainable, sum(trainability))
  
  # Compute mean & alpha
  for (i in c(1:compressed_dim)) {
    if (trainability[i]) {
      mean[i] <- mean(y_train[,i])/255
      alpha[i] <- sigmaRange*sd(y_train[,i])
    }
  }
  
  # Building new model
  for (i in c(1:compressed_dim)) {
    
    # Layer for output neuron i
    comp[[i]] <- layer_dense(x, units = 1, 
                             activation = activation_generator(i), 
                             input_shape = original_dim, name = sprintf('Output_neuron_%d', i),
                             trainable = trainability[i])
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
    epochs = epochs,
    batch_size = batch_size
  )
  
  # Evaluation
  evaluations[i+1] <- evaluate(fae, x_test, x_test)
  
  # Save weights
  save_model_weights_hdf5(fae, filepath = 'weights')
}

# -----------------------------------------------------------------------------------------------------------------
# Checking results

# Save Model
save_model_weights_hdf5(fae, filepath = 'models/FullAE_Id_SoftPlus_6_W_1')
save_model_weights_hdf5(cae, filepath = 'models/CompAE_Id_SoftPlus_6_W_1')

# Views on test data
y_train <- predict_on_batch(cae, x_train)
y_train <- apply(y_train, 2, round)
y_train_df <- data.frame(y_train)

# Distribution
for (i in c(0:15)) {
  plot(ggplot(melt(y_train_df[,(i*16 + 1):((i+1)*16)]), aes(x = value, fill = variable)) + geom_density(alpha = 0.2))
  ggsave(sprintf("Id_SoftPlus_6_%d_1.png", i))
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
  y[,i] <- (2^(entropyAE[i] - 8)) * cdf(y_train[,i])
}

# Checking the distribution
y_df <- data.frame(y)
for (i in c(0:15)) {
  plot(ggplot(melt(y_df[,(i*16 + 1):((i+1)*16)]), aes(x = value, fill = variable)) + geom_density(alpha = 0.2))
  ggsave(sprintf("Id_SoftPlus_6_%d_CDF_1.png", i))
}

# Check if encoded digit are different for each class
y_df$Label <- mnist$train$y
class_mean <- matrix(vector(mode = 'numeric', compressed_dim*10), nrow = compressed_dim)
class_sd <- matrix(vector(mode = 'numeric', compressed_dim*10), nrow = compressed_dim)
for (i in c(1:compressed_dim)) {
  for (j in c(1:10)) {
    class_mean[i,j] <- mean(y_df[which(y_df$Label==j), i])
    class_sd[i,j] <- sd(y_df[which(y_df$Label==j), i])
  }
}

# For each class, plot mean & standart deviation of neurons
for (i in c(1:10)) {
  plot(class_mean[,i])
  plot(class_sd[,i])
}