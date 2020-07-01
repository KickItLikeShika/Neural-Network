import numpy as np



def training(X, y, theta1, theta2):
    """Perform Forward propagation and backpropagation."""

    # preprocessing
    m = y.size

    # X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # Forward propagation
    Z2 = np.dot(X, theta1)
    A2 = sigmoid(Z2)
    # A2 = np.concatenate([np.ones((A2.shape[0], 1)), A2], axis=1)

    Z3 = np.dot(A2, theta2)
    A3 = sigmoid(Z3)


    # Backpropagation

    # error in output
    A3_error = y - A3 
    # applying derivative of sigmoid to error
    A3_delta = A3_error*sigmoid_gradient(A3) 


    z2_error = A3_delta.dot(theta2.T) # z2 error: how much our hidden layer weights contributed to output error
    z2_delta = z2_error*sigmoid_gradient(A2) # applying derivative of sigmoid to z2 error


    theta1 += X.T.dot(z2_delta) # adjusting first set (input --> hidden) weights
    theta2 += A2.T.dot(A3_delta) # adjusting second set (hidden --> output) weights


    # The final output for the neural network with the current parameters
    return A3



# def lossFunction(X, y, theta):
#     summation = 0
#     m = y.size

#     for i in range(m):
#         suqred_error = (trained([i]) - y) ** 2
#         summation += suqred_error
    
#     return summation


def sigmoid_gradient(z):
    """Get the G prime."""
    # Initialize the g zeros to save the results correctly
    g = np.zeros(z.shape)
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def sigmoid(z):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-z))


def main():
    # The featres
    X = np.array([[0, 0, 1], 
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    # The target
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # The cells = number of rows
    cells = 4

    m = y.size

    # We initialize the weights randomly with
    # numbers between zero and one

    # Design the neural network
    # Input
    input_layer = X.copy()
    # The weights of the input layer
    # We initialize it that way to help us later
    # (we don't have to transpose it)
    theta1 = np.random.rand(X.shape[1], cells)
    # The weihgts of hidden layer 
    theta2 = np.random.rand(4, 1)
    # The output
    output = np.zeros(y.shape)

    # Test
    print("X \n", X)
    print("Theta 1 \n", theta1)
    print("Y \n", y)
    print("Theta 2 \n", theta2)
    print("Output \n", output)


    # Train
    # train(input_layer, y, theta1, theta2, m)
    for i in range(10000):
        output = training(X, y, theta1, theta2)
        print("Output: \n", y)
        print('Predicted Output: \n', output)
        # print("Loss: ", J)


if __name__ == '__main__':
    main()