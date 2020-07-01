import numpy as np


def train(X, y, W1, W2):
    """Perform Forward propagation and backpropagation."""
    #forward propagation through our network
    Z2 = np.dot(X, W1) # dot product of X (input) and first set of 3x2 weights
    A2 = sigmoid(Z2) # activation function
    Z3 = np.dot(A2, W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = sigmoid(Z3) # final activation function

    # backward propagate through the network
    o_error = y - o # error in output
    o_delta = o_error*sigmoid_gradient(o) # applying derivative of sigmoid to error


    z2_error = o_delta.dot(W2.T) # z2 error: how much our hidden layer weights contributed to output error
    z2_delta = z2_error*sigmoid_gradient(A2) # applying derivative of sigmoid to z2 error


    W1 += X.T.dot(z2_delta) # adjusting first set (input --> hidden) weights
    W2 += A2.T.dot(o_delta) # adjusting second set (hidden --> output) weights

    return o


def sigmoid(z):
    """The sigmoid function."""
    # activation function
    return 1/(1+np.exp(-z))


def sigmoid_gradient(z):
    """Get the G prime."""
    #derivative of sigmoid
    return z * (1 - z)


def predict(xPredicted, X, y, W1, W2):
    print ("Predicted data based on trained weights: ")
    print ("Input (scaled): \n" + str(xPredicted))
    print ("Output: \n" + str(train(X, y, W1, W2)))


def main():

    # X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    y = np.array(([92], [86], [89]), dtype=float)
    xPredicted = np.array(([4,8]), dtype=float)


    # scale units
    #print(X)
    X = X/np.amax(X, axis=0) # maximum of X array
    #print(X)
    #parameters
    inputSize = 2
    outputSize = 1
    hiddenSize = 3


    #weights
    W1 = np.random.randn(inputSize, hiddenSize) # (2x3) weight matrix from input to hidden layer
    #    print(self.W1)
    W2 = np.random.randn(hiddenSize, outputSize) # (3x1) weight matrix from hidden to output layer
    #    print(self.W2)



    #print(xPredicted)
    xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
    #print(xPredicted)
    y = y/100 # max test score is 100
    for i in range(3000): # trains the NN 1,000 times
        print ("# " + str(i) + "\n")
        print ("Input (scaled): \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(train(X, y, W1, W2)))
        print ("Loss: \n" + str(np.mean(np.square(y - train(X, y, W1, W2)))) )# mean sum squared loss
        print ("\n")

    predict(xPredicted, X, y, W1, W2)

if __name__ == '__main__':
    main()