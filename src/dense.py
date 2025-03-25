# this is our most basic functionalliy for a nn. We wish to show that we can do the basic nn applications
# like handwriting analysis using our library

import numpy as np # eventually this will be made obsolete by our own math libraries
import time

class NeuralNetwork:

    """
    this is just a 3 layer nn. Each layer has a certain dimension. 

    :param int input_dim: Number of input neurons.
    :param int hidden_dim: Number of neurons in the hidden layer.
    :param int output_dim: Number of output neurons.
    :param float learning_rate: Learning rate for training.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # We will adjust the following with our own weight initializations later 
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

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

    def relu(self, z, leaky = 0):
        """
        Computes the relu activation function.

        :param np.ndarray z: The input array.
        :param float leaky: The value of alpha in our relu function.
        :return: The result of applying relu on z.
        :rtype: np.ndarray
        """
        if z > 0:
            return z
        else:
            return leaky * z
    
    def relu_derivitive(self, a, leaky = 0):
        """
        Computes the derivative of the relu function.

        :param np.ndarray a: The output of the relu function.
        :return: The derivative of the relu function.
        :rtype: np.ndarray
        """
        if a > 0:
            return 1
        else:
            return leaky

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
        # Compute the input for the hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Compute the output layer input and activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """
        Performs backward propagation and update the network's weights and biases.

        :param np.ndarray X: Input data of shape (n_samples, input_dim).
        :param np.ndarray y: True labels of shape (n_samples, output_dim).
        :param np.ndarray output: Output from the forward propagation of shape (output_dim).
        """
        # Calculate the error in the output layer
        error_output = y - output
        delta_output = error_output * self.sigmoid_derivative(output)

        # Calculate the error in the hidden layer
        error_hidden = np.dot(delta_output, self.W2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.a1) # any good caching tricks I can use here... Seems my foundations need waxing...

        # Update weights and biases for hidden-output layer
        self.W2 += np.dot(self.a1.T, delta_output) * self.learning_rate
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate

        # Update weights and biases for input-hidden layer
        self.W1 += np.dot(X.T, delta_hidden) * self.learning_rate
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

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
        nn = NeuralNetwork(input_dim=2, hidden_dim=20, output_dim=1, learning_rate=0.1)

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
