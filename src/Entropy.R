# Charge database
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# AE
cae <- load_model_hdf5('CompAE_0.001')
y_train <- predict_on_batch(cae, x_train)

# PCA
x_pca <- prcomp(x_train, center = TRUE, scale = FALSE)

# Ticking
y_train <- apply(y_train, 2, round)
x_pca$x <- apply(x_pca$x, 2, round)

# Compute entropy
entropy <- c(1:784)
for (i in c(1:784)) {
  e <- entropy(table(x_train[,i]), unit = 'log2')
  if (is.na(e)) {
    entropy[i] <- 0
  } else {
    entropy[i] <- e
  }
}

# Compute entropy AE
entropyAE <- c(1:256)
for (i in c(1:256)) {
  e <- entropy(table(y[,i]), unit = 'log2')
  if (is.na(e)) {
    entropyAE[i] <- 0
  } else {
    entropyAE[i] <- e
  }
}

# Compute entropy PCA
entropyPCA <- c(1:784)
for (i in c(1:784)) {
  e <- entropy(table(x_pca$x[,i]), unit = 'log2')
  if (is.na(e)) {
    entropyPCA[i] <- 0
  } else {
    entropyPCA[i] <- e
  }
}

# Histograms
y_df <- data.frame(y_train/255)
for (i in c(1:90)) {
  ggplot(melt(y_df[,i]), aes(x = value)) + geom_histogram(alpha = 0.5)
  ggsave(sprintf("AE%d.png", i))
}

print(sum(entropy))
entropy <- c(entropy, sum(entropy))

print(sum(entropyAE))
entropyAE <- c(entropyAE, sum(entropyAE))

print(sum(entropyPCA))
entropyPCA <- c(entropyPCA, sum(entropyPCA))