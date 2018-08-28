# Model to encode MNIST Data base with 8 bit entropy and uniform distribution
library(keras)
library(ggplot2)
library(reshape2)
library(entropy)
library(dplyr)

# Parameters
original_dim <- 784L
compressed_dim <- 128L
batch_size <- 50L
epochs <- 20L

# Preparing data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# -----------------------------------------------------------------------------------------------------------------
# Initialization of the ANN

# Model definition
x <- layer_input(shape = c(original_dim), name = 'Input')
compress_1 <- layer_dense(x, units = 256, activation = 'relu', name = 'First_Compression')
compress_2 <- layer_dense(compress_1, units = 128, activation = 'relu', name = 'Final_Compression')
decompress_1 <- layer_dense(compress_2, units = 256, activation = activation_softplus, name = 'First_Decompression')
decompress_2 <- layer_dense(decompress_1, units = original_dim, activation = activation_softplus, name = 'Final_Output')

# Compile
cae <- keras_model(x, compress_2)
fae <- keras_model(x, decompress_2)
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

# Add to the stories
plot(hist)

# Views on test data
y_train <- predict_on_batch(cae, x_train)
y_train <- apply(y_train, 2, round)
y_train_df <- data.frame(y_train)

# Distribution
for (i in c(0:7)) {
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
