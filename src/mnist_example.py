from mnist import MNIST
import random

# this is the naive side pass class


# this is our most basic functionalliy for a nn. We wish to show that we can do the basic nn applications
# like handwriting analysis using our library
# this is a library to learn the basics of nn's and how they work. 

import numpy as np # eventually this will be made obsolete by our own math libraries
import time
from initializers.initalizers import init_weights

class NaiveSideNet:

    """
    this is just a 3 layer nn. Each layer has a certain dimension. 

    :param int input_dim: Number of input neurons.
    :param int hidden_dim: Number of neurons in the hidden layer.
    :param int output_dim: Number of output neurons.
    :param float learning_rate: Learning rate for training.
    """

    def __init__(self, dim_in, dim_out, height_hidden, width_hidden, side_layers = [], activation_function = "sigmoid", learning_rate=0.01, learning_rate_decay = 0.):
        
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        # feedforward layers
        self.height_hidden = height_hidden # the dimension of each layer for each layer in network (assumes all values are ints >0)
        self.width_hidden = width_hidden # the dimension of each layer for each layer in network (assumes all values are ints >0)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.depth = width_hidden + 2
        self.weights = []
        self.biases = []
        self.side_pass = False
        # sidepass layers
        if len(side_layers) > 0:
            self.side_pass = True
            self.side_layers = [self.height_hidden * self.width_hidden] + side_layers
            self.side_depth = len(side_layers) 
            self.side_weights = []
            self.side_biases = []

        self.init_technique = "xavier"

        if activation_function == "tanh":
            self.activation_function = self.tanh
            self.activation_function_derivative = self.tanh_derivitive  
        elif activation_function == "relu":
            self.activation_function = self.relu
            self.activation_function_derivative = self.relu_derivitive
            self.init_technique = "he"
        elif activation_function == "softmax":
            self.activation_function = self.softmax
            self.activation_function_derivative = self.softmax_derivitive
        elif activation_function == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_function_derivative = self.sigmoid_derivative
        else:
            raise ValueError("Invalid activation function")

        # --- Initialize weights and biases --- 
        # feedforward weights and biases
        # TODO: Be better!!! make a array of dimensions and init from there. 
        self.weights.append(init_weights(self.dim_in, self.height_hidden, self.init_technique))
        self.biases.append(np.zeros((1, self.height_hidden)))
        for i in range(self.depth - 3):
            self.weights.append(init_weights(self.height_hidden, self.height_hidden, self.init_technique))
            self.biases.append(np.zeros((1, self.height_hidden)))
        self.weights.append(init_weights(self.height_hidden, self.dim_out, self.init_technique))
        self.biases.append(np.zeros((1, self.dim_out)))

        # sidepass weights and biases
        if len(side_layers) > 0:
            for i in range(1, self.side_depth):
                self.side_weights.append(init_weights(side_layers[i-1], side_layers[i], self.init_technique))
                self.side_biases.append(np.zeros((1, side_layers[i])))

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
    
    def softmax(self, z, T = 1.):
        """
        Computes the softmax activation function.
        Numerically stabalized.
        
        :param np.ndarray z: The input array.
        :return: The result of applying softmax on z.
        :rtype: np.ndarray
        """
        z = z/T
        shiftx = z - np.max(z, axis=1, keepdims=True)
        exp_x = np.exp(shiftx)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        softmax = exp_x / sum_exp_x
        return softmax
    
    def softmax_derivative(self, a, T = 1.):
        """
        Computes the derivative of the softmax function.

        :param np.ndarray a: The output of the softmax function.
        :return: The derivative of the softmax function.
        :rtype: np.ndarray
        """

        return a/T * (1 - a)
    
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
        self.outputs = [X] # L1
        for i in range(self.depth - 2): # L2,..,Ln-1
            self.outputs.append(self.activation_function(np.dot(self.outputs[-1], self.weights[i]) + self.biases[i]))
        self.outputs.append(self.softmax(np.dot(self.outputs[-1], self.weights[-1]) + self.biases[-1])) # Ln

        if self.side_pass:
            all_outputs = np.array([])
            for output in self.outputs: 
                for elm in output: 
                    all_outputs = np.append(all_outputs, elm)

            self.side_outputs = [all_outputs]
            for i in range(self.side_depth - 1):
                self.side_outputs.append(self.activation_function(np.dot(self.side_outputs[-1], self.side_weights[i]) + self.side_biases[i]))
            print("side_outputs", self.side_outputs) #(1,32,2)[[(,),(,),,,,,,,,,,...]]

        # return self.side_outputs[-1] if side_pass else self.outputs[-1]

    def backward(self, y, no_side_gradients = False):
        """
        Performs backward propagation and update the network's weights and biases.

        :param np.ndarray y: True labels of shape (n_samples, output_dim).
        :param np.ndarray output: Output from the forward propagation of shape (output_dim).
        """

        if self.side_pass:
            side_deltas = [None] * (self.side_depth - 1)
            side_deltas[-1] = (y - np.array(self.side_outputs[-1])) * self.softmax_derivative(np.array(self.side_outputs[-1]))
            for i in range(self.side_depth - 2, 0, -1):
                side_deltas.append(np.dot(side_deltas[-1], self.side_weights[i].T) * self.activation_function_derivative(np.array(self.side_outputs[i])))
                
            if not no_side_gradients:
                for i in range(self.side_depth - 1): 
                    self.side_weights[-(i+1)] += np.dot(np.array(self.side_outputs[-(i+2)]).T, side_deltas[i]) * self.learning_rate
                    self.side_biases[-(i+1)] += np.sum(side_deltas[i], axis=0, keepdims=True) * self.learning_rate
            
        deltas = [None] * (self.depth - 1)
        deltas[-1] = (side_deltas[-1] if self.side_pass else y - np.array(self.outputs[-1])) * self.softmax_derivative(np.array(self.outputs[-1]))

        # Propagate the error backwards
        for i in range(self.depth - 2, 0, -1):
            deltas[i-1] = np.dot(deltas[i], self.weights[i].T) * self.activation_function_derivative(np.array(self.outputs[i]))

        # Update weights and biases
        for i in range(self.depth - 1):
            self.weights[i] += np.dot(np.array(self.outputs[i]).T, deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate

    def deterministic_choose(self, X):
        """
        Chooses the class of the input.
        """
        self.forward(X)
        outputs = self.side_outputs[-1] if self.side_pass else self.outputs[-1]
        choices = np.zeros((len(outputs), 10))
        in_loop = range(len(outputs[0]))
        for i, output in enumerate(outputs):
            max = 0
            for j in in_loop:       
                if output[j] > max:
                    max = output[j]
                    max_index = j
            choices[i][max_index] = 1
        print("choices", choices)
        return choices
    
    def stochastic_choose(self, X):
        """
        Chooses the class of the input.
        """
        self.forward(X)
        outputs = self.side_outputs[-1] if self.side_pass else self.outputs[-1]
        choices = []
        for output in outputs:
            choices.append(np.random.choice(len(output), p=output))
        return choices
    def train(self, X, y, epochs=10000):
        """
        Train the neural network using the provided data.

        :param np.ndarray X: Input data of shape (n_samples, input_dim).
        :param np.ndarray y: True labels of shape (n_samples, output_dim).
        :param int epochs: Number of training iterations.
        """
        '''tick = 0
        avg_loss = 0'''
        for epoch in range(1,epochs+1):
            self.learning_rate *= (1 - self.learning_rate_decay)
            self.forward(X)
            print("forward done")
            self.backward(y)
            print("epoch", epoch, "done")
            # TODO: change this to a more appropriate loss function i.e. cross entropy
            '''loss = np.mean((y - output) ** 2)
            avg_loss += loss
            # Print loss every so many epochs
            if epoch % (epochs/10) == 0:

                # print(f'Epoch {epoch}, Loss: {loss:.3f}')
                if tick == 0:
                    tick = 1
                    avg_loss_old = avg_loss
                elif avg_loss > avg_loss_old - .001: # we have converged #NOTE: .001 is hardcoded for now. 
                    break
                else:
                    avg_loss_old = avg_loss

                avg_loss = 0'''


if __name__ == "__main__":
    tries = 10
    failures = 0
    
    # Initialize MNIST data loader
    mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/MNIST/raw')

    # Load the training data
    images, labels = mndata.load_training()
    labels = np.array([[1 if labels[i] == j else 0 for j in range(10)] for i in range(len(labels))])

    test_images, test_labels = mndata.load_testing()
    test_labels = np.array([[1 if test_labels[i] == j else 0 for j in range(10)] for i in range(len(test_labels))])
    print("labels", labels)
    print("test labels", test_labels)
    start_time = time.perf_counter()
    print("start time", start_time)
    for i in range(tries):

        np.random.seed(1200+i) # this actually only works well under certain initial weights. We need to be able to create a general working model. 

        # make a NeuralNetwork instance with 2 input values, 2 hidden neurons, and 1 output value
        nn = NaiveSideNet(784, 10, 264, 2,  activation_function = "relu", learning_rate=0.1)
        print("nn made")
        # train our network
        nn.train(images, labels, epochs=1)
        print("full train done")

        # test our trained network
        output = nn.deterministic_choose(test_images)
        if output != test_labels:
            failures += 1
            print("We failed. failures = ", failures, "output", output, "should be", test_labels)

    end_time = time.perf_counter()
    print("accuracy:", (tries-failures)/tries, "\naverage time (seconds): ", (end_time - start_time)/tries) # time includes train/test time for each weight initialization

'''# Initialize MNIST data loader
mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/MNIST/raw')

# Load the training data
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Get a random index
index = random.randrange(0, len(images)) 

# Display the image
print(mndata.display(images[index])) '''