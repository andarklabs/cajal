# this is our most basic functionalliy for a nn. We wish to show that we can do the basic nn applications
# like handwriting analysis using our library

import numpy as np # eventually this will be made obsolete by our own math libraries

class NeuralNetwork:

    """
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

    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.

        :param np.ndarray z: The input array.
        :return: The result of applying sigmoid on z.
        :rtype: np.ndarray
        """
        return 1. / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """
        Compute the derivative of the sigmoid function.

        :param np.ndarray a: The output of the sigmoid function.
        :return: The derivative of the sigmoid function.
        :rtype: np.ndarray
        """
        return a * (1 - a)

    def forward(self, X):
        """
        Perform forward propagation.

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
        Perform backward propagation and update the network's weights and biases.

        :param np.ndarray X: Input data of shape (n_samples, input_dim).
        :param np.ndarray y: True labels of shape (n_samples, output_dim).
        :param np.ndarray output: Output from the forward propagation.
        """
        # Calculate the error in the output layer
        error_output = y - output
        delta_output = error_output * self.sigmoid_derivative(output)

        # Calculate the error in the hidden layer
        error_hidden = np.dot(delta_output, self.W2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.a1)

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
        for epoch in range(epochs):
            # Forward propagation step
            output = self.forward(X)
            # Backward propagation step
            self.backward(X, y, output)
            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean((y - output) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.3f}')

if __name__ == "__main__":
    np.random.seed(1) # this actually only works well under certain initial weights. We need to be able to create a general working model. 

    # Example usage: Solving the XOR problem
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # Create a NeuralNetwork instance with 2 input neurons, 2 hidden neurons, and 1 output neuron
    nn = NeuralNetwork(input_dim=2, hidden_dim=2, output_dim=1, learning_rate=0.1)
    nn.train(X, y, epochs=10000)

    # Test the trained network
    for sample in X:
        output = nn.forward(np.array([sample]))
        print(f'Input: {sample}, Output: {output[0][0]:.3f}')
