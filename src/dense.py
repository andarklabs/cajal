# this is our most basic functionalliy for a nn. We wish to show that we can do the basic nn applications
# like handwriting analysis using our library

import numpy as np # eventually this will be made obsolete by our own math libraries
import time
from initializers.initalizers import init_weights

class NeuralNetwork:

    """
    this is just a 3 layer nn. Each layer has a certain dimension. 

    :param int input_dim: Number of input neurons.
    :param int hidden_dim: Number of neurons in the hidden layer.
    :param int output_dim: Number of output neurons.
    :param float learning_rate: Learning rate for training.
    """

    def __init__(self, layers, activation_function = "sigmoid", learning_rate=0.01):
        self.layers = layers # the dimension of each layer for each layer in network (assumes all values are ints >0)
        self.learning_rate = learning_rate
        self.depth = len(layers)
        self.weights = []
        self.biases = []
        if activation_function == "tanh":
            self.activation_function = self.tanh
            self.activation_function_derivative = self.tanh_derivitive  
        elif activation_function == "relu":
            self.activation_function = self.relu
            self.activation_function_derivative = self.relu_derivitive
        elif activation_function == "softmax":
            self.activation_function = self.softmax
            self.activation_function_derivative = self.softmax_derivitive
        elif activation_function == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_function_derivative = self.sigmoid_derivative
        else:
            raise ValueError("Invalid activation function")

        # DONE: We will adjust the following with our own weight initializations later 
        for i in range(1, self.depth):
            self.weights.append(init_weights(layers[i-1], layers[i]))
            self.biases.append(np.zeros((1, layers[i])))

    def tanh(self, z):
        """
        Computes the tanh activation function.

        :param np.ndarray z: The input array.
        :return: The result of applying tanh on z.
        :rtype: np.ndarray
        """
        return 2./(1+np.exp(-2*z)) - 1

    def tanh_derivitive(self, a):
        """
        Computes the derivative of the tanh function.

        :param np.ndarray a: The output of the tanh function.
        :return: The derivative of the tanh function.
        :rtype: np.ndarray
        """
        return 1 - pow(a,2)

    def relu(self, z, leaky = 0.01):
        """
        Computes the relu activation function.

        :param np.ndarray z: The input array.
        :param float leaky: The value of alpha in our relu function.
        :return: The result of applying relu on z.
        :rtype: np.ndarray
        """
        result = np.zeros(z.shape)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if z[i,j] > 0:
                    result[i,j] = z[i,j]
                else:
                    result[i,j] = leaky * z[i,j]
        return result
    
    def relu_derivitive(self, a, leaky = 0.01):
        """
        Computes the derivative of the relu function.

        :param np.ndarray a: The output of the relu function.
        :return: The derivative of the relu function.
        :rtype: np.ndarray
        """
        result = np.zeros(a.shape)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i,j] > 0:
                    result[i,j] = 1
                else:
                    result[i,j] = leaky
        return result
    def softmax(self, z):
        """
        Computes the softmax activation function.
        Numerically stabalized.
        
        :param np.ndarray z: The input array.
        :return: The result of applying softmax on z.
        :rtype: np.ndarray
        """
        shiftz = z - np.max(z)
        exps = np.exp(shiftz)
        return exps/np.sum(exps)

    def softmax_derivitive(self, a):
        """
        Computes the derivative of the softmax function.

        :param np.ndarray a: The output of the softmax function.
        :return: The derivative of the softmax function.
        :rtype: np.ndarray
        """
        return -np.outer(a,a) + np.diag(a.flatten())    

    def sigmoid(self, z):
        """
        Computes the sigmoid activation function.

        :param np.ndarray z: The input array.
        :return: The result of applying sigmoid on z.
        :rtype: np.ndarray
        """
        return 1. / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """
        Computes the derivative of the sigmoid function.

        :param np.ndarray a: The output of the sigmoid function.
        :return: The derivative of the sigmoid function.
        :rtype: np.ndarray
        """
        return a * (1 - a)

    def forward(self, X):
        """
        Performs forward propagation.

        :param np.ndarray X: Input data of shape (n_samples, input_dim).
        :return: The output of the network.
        :rtype: np.ndarray
        """
        # store the outputs of each layer in an array to use in backprop 
        self.outputs = [X]
        for i in range(self.depth - 1):
            self.outputs.append(self.activation_function(np.dot(self.outputs[-1], self.weights[i]) + self.biases[i]))

        return self.outputs[-1]

    def backward(self, X, y, output):
        """
        Performs backward propagation and update the network's weights and biases.

        :param np.ndarray X: Input data of shape (n_samples, input_dim).
        :param np.ndarray y: True labels of shape (n_samples, output_dim).
        :param np.ndarray output: Output from the forward propagation of shape (output_dim).
        """
        # Calculate the error in the each layer and put it (in constant time) at the end of each array (so the error is in reverse of the layers)
        error_layers = [y - output]
        delta_layers = [error_layers[-1] * self.activation_function_derivative(output)]
        for i in range(self.depth - 2, 0, -1):
            # Calculate the error in each layer
            error_layers.append(np.dot(delta_layers[-1], self.weights[i].T))
            delta_layers.append(error_layers[-1] * self.activation_function_derivative(self.outputs[i]))

        for i in range(self.depth - 1):
            # Update weights and biases for each ff layer
            self.weights[-(i+1)] += np.dot(self.outputs[-(i+2)].T, delta_layers[i]) * self.learning_rate
            self.biases[-(i+1)] += np.sum(delta_layers[i], axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        """
        Train the neural network using the provided data.

        :param np.ndarray X: Input data of shape (n_samples, input_dim).
        :param np.ndarray y: True labels of shape (n_samples, output_dim).
        :param int epochs: Number of training iterations.
        """
        tick = 0
        for epoch in range(1,epochs+1):

            output = self.forward(X)
            self.backward(X, y, output)
            
            # Print loss every 1000 epochs
            if epoch % 1000 == 0:

                loss = np.mean((y - output) ** 2)
                # print(f'Epoch {epoch}, Loss: {loss:.3f}')
                if tick == 0:
                    tick = 1
                    loss_old = loss
                elif loss > loss_old - .001: # we have converged
                    break
                else:
                    loss_old = loss


if __name__ == "__main__":
    tries = 100
    failed = 0
    start_time = time.perf_counter()

    # the XOR problem
    X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
    y = np.array([[0],
                [1],
                [1],
                [0]])

    for i in range(tries):

        np.random.seed(200+i) # this actually only works well under certain initial weights. We need to be able to create a general working model. 

        # make a NeuralNetwork instance with 2 input values, 2 hidden neurons, and 1 output value
        nn = NeuralNetwork([2, 4, 1], activation_function = "relu", learning_rate=0.1)

        # train our network
        nn.train(X, y, epochs=10000)

        # test our trained network
        loss = 0
        for sample in X:
            output = nn.forward(np.array([sample]))[0][0]
            loss += pow(((sample[0]+sample[1])%2) - output,2) # MSE of our problem
            # print(f'Input: {sample}, Output: {output:.3f}')

        if loss > .10: # we fail to converge properly
            failed += 1
            print("We failed. loss = ", loss, "output", [nn.forward(np.array([sample]))[0][0] for sample in X]) # we can look at the outputs where we failed

    end_time = time.perf_counter()
    print("accuracy:", (tries-failed)/tries, "\naverage time (seconds): ", (end_time - start_time)/tries) # time includes train/test time for each weight initialization
