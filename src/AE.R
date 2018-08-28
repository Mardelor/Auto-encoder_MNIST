# Giving a vector of layers' sizes, initialize the network (weights)
#
# returns the network (a layers' list)
initAE <- function(sizes) {
    nLayers <- length(sizes)
    auto_encoder <- list()
    # Network is represented here by a list of layers
    print(sprintf("Number of layers : %d", nLayers))
    print(sprintf("Size : %d", sizes))
    for (i in c(1:(nLayers-1))) {
        # A layer is a list :
        #   1. Matrix of weights, randomly initialize
        #   2. Output vector
        inSize = sizes[i]
        outSize = sizes[i+1]
        print(inSize)
        print(outSize)
        layer <- list(
        matrix(runif((inSize+1)*outSize, 0, 1) - 0.5, outSize, inSize + 1),
        vector("numeric", outSize))
        auto_encoder[[i]] <- layer
    }
    return(auto_encoder)
}

#AE init
auto_encoder <- initAE(c(784, 256, 784))

# ReLu activation function
ReLu <- function(value) {
    return(max(0, value))
}

# Giving an input, compute the output of the network
#
# returns the output vector
feedforward <- function(input) {
    if ((ncol(auto_encoder[[1]][[1]])) != length(input) + 1) {
        print("Error : input Data does not fit to the network")
        return(NA)
    }
    input <- unname(unlist(input))
    inp <- c(input, 1)
    for (layer in auto_encoder) {
        # Using ReLu activation function
        layer[[2]] <- sapply(layer[[1]] %*% inp, ReLu)
        inp <- c(layer[[2]], 1)
    }
    return(inp[1:(length(inp)-1)])
}

# Derivative ReLu
dReLu <- function(value) {
    if (value > 0) {
        return(1)
    }
    return(0)
}

# Stochastic Gradient Descent algorithm
#
# Giving training data, the numbers of epochs, the size of each sample,
# the learning rate and the testing data, it trains the network
SGD <- function(trainingData, nEpochs, sampleSize, step, testData) {
    for (i in c(1:nEpochs)) {
        # Shuffle rows (images)
        trainWritable <- trainingData[sample(nrow(trainingData)),]
        for (j in c(1:(nrow(trainingData)/sampleSize))) {
            beg <- (j-1)*sampleSize + 1
            end <- j*sampleSize
            sample <- trainWritable[c(beg:end),]
            updateWeights(sample, step)
        }
    }
}

# Giving a sample of points, train the network
updateWeights <- function(sample, lRate) {
    nablaW <- list()
    # Compute derivative (list of weight matrices)
    for (j in c(1:length(sample[,1]))) {
        point <- sample[j,]
        dNablaW <- backProp(unname(unlist(point)))
        if(length(nablaW) == 0) {
            for (i in c(1:(length(dNablaW)))) {
                nablaW[[i]] <- dNablaW[[i]]
            }
        }
        else {
            for (i in c(1:(length(dNablaW)))) {
                nablaW[[i]] <- nablaW[[i]] + dNablaW[[i]]
            }
        }
    }
    # Update network's weights
    for (i in c(1:length(auto_encoder))) {
        print(i)
        auto_encoder[[i]][[1]] <<- (auto_encoder[[i]][[1]] - lRate/length(sample)*nablaW[[i]])
    }
}

# Backpropagation algorithm
backProp <- function(point) {
    as <- list()
    zs <- list()
    errors <- list()
    dWeight <- list()
    L <- length(auto_encoder)

    inp <- c(point, 1)
    i = 1
    for (layer in auto_encoder) {
        # Compute z and a
        zs[[i]] <- layer[[1]] %*% inp
        as[[i]] <- sapply(zs[[i]], ReLu)
        inp <- c(as[[i]], 1)
        i <- i+1
    }

    # Compute the error
    dCost <- 2*(as[[L]] - point)
    errors[[L]] <- dCost*sapply(zs[[L]], dReLu)
    dWeight[[L]] <- errors[[L]] %*% t(c(as[[L-1]], 1))

    # BackPropagation
    if(L > 2) {
        for (l in c((L-1):2)) {
            errors[[l]] <- (t(auto_encoder[[l+1]][[1]][,1:(ncol(auto_encoder[[l+1]][[1]])-1)]) %*% errors[[l+1]])*sapply(zs[[l]], dReLu)
            dWeight[[l]] <- errors[[l]] %*% t(c(as[[l-1]], 1))
        }
    }
    errors[[1]] <- (t(auto_encoder[[2]][[1]][,1:(ncol(auto_encoder[[2]][[1]])-1)]) %*% errors[[2]])*sapply(zs[[1]], dReLu)
    dWeight[[1]] <- errors[[1]] %*% t(c(point, 1))

    # dWeight now contains matrix of gradients in proper order
    return(dWeight)
}
