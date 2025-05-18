from mnist import MNIST
import random
import torch
import time

# this is the naive side pass class


# this is our most basic functionalliy for a nn. We wish to show that we can do the basic nn applications
# like handwriting analysis using our library
# this is a library to learn the basics of nn's and how they work. 

def init_weights(in_dim, out_dim, technique="xavier"):
    """Initialize weights based on the specified technique."""
    if technique == "xavier":
        return torch.nn.init.xavier_uniform_(torch.empty(in_dim, out_dim))
    elif technique == "he":
        return torch.nn.init.kaiming_uniform_(torch.empty(in_dim, out_dim), nonlinearity='relu')
    else:
        return torch.randn(in_dim, out_dim) * 0.01

class NaiveSideNet:

    """
    this is just a 3 layer nn. Each layer has a certain dimension. 

    :param int input_dim: Number of input neurons.
    :param int hidden_dim: Number of neurons in the hidden layer.
    :param int output_dim: Number of output neurons.
    :param float learning_rate: Learning rate for training.
    """

    def __init__(self, dim_in, dim_out, height_hidden, width_hidden, side_layers = [], activation_function = "sigmoid", learning_rate=0.01, learning_rate_decay = 0., device=None):
        
        # Set device - use MPS if available, otherwise CPU
        self.device = device if device is not None else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
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
        self.weights.append(init_weights(self.dim_in, self.height_hidden, self.init_technique).to(self.device))
        self.biases.append(torch.zeros(1, self.height_hidden, device=self.device))
        for i in range(self.depth - 3):
            self.weights.append(init_weights(self.height_hidden, self.height_hidden, self.init_technique).to(self.device))
            self.biases.append(torch.zeros(1, self.height_hidden, device=self.device))
        self.weights.append(init_weights(self.height_hidden, self.dim_out, self.init_technique).to(self.device))
        self.biases.append(torch.zeros(1, self.dim_out, device=self.device))

        # sidepass weights and biases
        if len(side_layers) > 0:
            for i in range(1, self.side_depth):
                self.side_weights.append(init_weights(side_layers[i-1], side_layers[i], self.init_technique).to(self.device))
                self.side_biases.append(torch.zeros(1, side_layers[i], device=self.device))

    def tanh(self, z):
        """
        Computes the tanh activation function.

        :param torch.Tensor z: The input tensor.
        :return: The result of applying tanh on z.
        :rtype: torch.Tensor
        """
        return torch.tanh(z)

    def tanh_derivitive(self, a):
        """
        Computes the derivative of the tanh function.

        :param torch.Tensor a: The output of the tanh function.
        :return: The derivative of the tanh function.
        :rtype: torch.Tensor
        """
        return 1 - a.pow(2)

    def relu(self, z, leaky = 0.01):
        """
        Computes the relu activation function.

        :param torch.Tensor z: The input tensor.
        :param float leaky: The value of alpha in our relu function.
        :return: The result of applying relu on z.
        :rtype: torch.Tensor
        """
        return torch.nn.functional.leaky_relu(z, leaky)
    
    def relu_derivitive(self, a, leaky = 0.01):
        """
        Computes the derivative of the relu function.

        :param torch.Tensor a: The output of the relu function.
        :return: The derivative of the relu function.
        :rtype: torch.Tensor
        """
        return torch.where(a > 0, torch.ones_like(a), torch.ones_like(a) * leaky)
    
    def softmax(self, z, T = 1.):
        """
        Computes the softmax activation function.
        Numerically stabalized.
        
        :param torch.Tensor z: The input tensor.
        :return: The result of applying softmax on z.
        :rtype: torch.Tensor
        """
        z = z/T
        return torch.nn.functional.softmax(z, dim=1)
    
    def softmax_derivative(self, a, T = 1.):
        """
        Computes the derivative of the softmax function.

        :param torch.Tensor a: The output of the softmax function.
        :return: The derivative of the softmax function.
        :rtype: torch.Tensor
        """
        return a/T * (1 - a)
    
    def sigmoid(self, z):
        """
        Computes the sigmoid activation function.

        :param torch.Tensor z: The input tensor.
        :return: The result of applying sigmoid on z.
        :rtype: torch.Tensor
        """
        return torch.sigmoid(z)
    
    def sigmoid_derivative(self, a):
        """
        Computes the derivative of the sigmoid function.

        :param torch.Tensor a: The output of the sigmoid function.
        :return: The derivative of the sigmoid function.
        :rtype: torch.Tensor
        """
        return a * (1 - a)
    
    def forward(self, X):
        """
        Performs forward propagation.

        :param torch.Tensor X: Input tensor of shape (n_samples, input_dim).
        :return: The output of the network.
        :rtype: torch.Tensor
        """
        # store the outputs of each layer in an array to use in backprop 
        self.outputs = [X] # L1
        for i in range(self.depth - 2): # L2,..,Ln-1
            self.outputs.append(self.activation_function(torch.matmul(self.outputs[-1], self.weights[i]) + self.biases[i]))
        self.outputs.append(self.softmax(torch.matmul(self.outputs[-1], self.weights[-1]) + self.biases[-1])) # Ln

        if self.side_pass:
            all_outputs = torch.cat([output.flatten() for output in self.outputs])
            
            self.side_outputs = [all_outputs]
            for i in range(self.side_depth - 1):
                self.side_outputs.append(self.activation_function(torch.matmul(self.side_outputs[-1], self.side_weights[i]) + self.side_biases[i]))
            print("side_outputs", self.side_outputs)

    def backward(self, y, no_side_gradients = False):
        """
        Performs backward propagation and update the network's weights and biases.

        :param torch.Tensor y: True labels tensor of shape (n_samples, output_dim).
        :param torch.Tensor output: Output from the forward propagation tensor of shape (output_dim).
        """
        if self.side_pass:
            side_deltas = [None] * (self.side_depth - 1)
            side_deltas[-1] = (y - self.side_outputs[-1]) * self.softmax_derivative(self.side_outputs[-1])
            for i in range(self.side_depth - 2, 0, -1):
                side_deltas.append(torch.matmul(side_deltas[-1], self.side_weights[i].t()) * self.activation_function_derivative(self.side_outputs[i]))
                
            if not no_side_gradients:
                for i in range(self.side_depth - 1): 
                    self.side_weights[-(i+1)] += torch.matmul(self.side_outputs[-(i+2)].t(), side_deltas[i]) * self.learning_rate
                    self.side_biases[-(i+1)] += side_deltas[i].sum(dim=0, keepdim=True) * self.learning_rate
            
        deltas = [None] * (self.depth - 1)
        deltas[-1] = (side_deltas[-1] if self.side_pass else y - self.outputs[-1]) * self.softmax_derivative(self.outputs[-1])

        # Propagate the error backwards
        for i in range(self.depth - 2, 0, -1):
            deltas[i-1] = torch.matmul(deltas[i], self.weights[i].t()) * self.activation_function_derivative(self.outputs[i])

        # Update weights and biases
        for i in range(self.depth - 1):
            self.weights[i] += torch.matmul(self.outputs[i].t(), deltas[i]) * self.learning_rate
            self.biases[i] += deltas[i].sum(dim=0, keepdim=True) * self.learning_rate

    def deterministic_choose(self, X):
        """
        Chooses the class of the input.
        """
        self.forward(X)
        outputs = self.side_outputs[-1] if self.side_pass else self.outputs[-1]
        choices = torch.zeros((len(outputs), 10), device=self.device)
        
        max_indices = torch.argmax(outputs, dim=1)
        for i, max_index in enumerate(max_indices):
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
            choices.append(torch.multinomial(output, 1).item())
        return choices
    def train(self, X, y, epochs=10000):
        """
        Train the neural network using the provided data.

        :param torch.Tensor X: Input tensor of shape (n_samples, input_dim).
        :param torch.Tensor y: True labels tensor of shape (n_samples, output_dim).
        :param int epochs: Number of training iterations.
        """
        for epoch in range(1,epochs+1):
            self.learning_rate *= (1 - self.learning_rate_decay)
            self.forward(X)
            print("forward done")
            self.backward(y)
            print("epoch", epoch, "done")


if __name__ == "__main__":
    tries = 10
    failures = 0
    
    # Initialize MNIST data loader
    mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/MNIST/raw')

    # Load the training data
    images, labels = mndata.load_training()
    
    # Convert to PyTorch tensors and move to MPS device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Convert images to normalized float tensors
    images = torch.tensor(images, dtype=torch.float32, device=device) / 255.0
    
    # One-hot encode labels
    label_indices = torch.tensor([labels], device=device).t()
    labels = torch.zeros(len(labels), 10, device=device)
    labels.scatter_(1, label_indices, 1)

    # Test data
    test_images, test_labels = mndata.load_testing()
    test_images = torch.tensor(test_images, dtype=torch.float32, device=device) / 255.0
    test_label_indices = torch.tensor([test_labels], device=device).t()
    test_labels = torch.zeros(len(test_labels), 10, device=device)
    test_labels.scatter_(1, test_label_indices, 1)
    
    print("labels", labels)
    print("test labels", test_labels)
    start_time = time.perf_counter()
    print("start time", start_time)
    for i in range(tries):
        torch.manual_seed(1200+i)

        # make a NeuralNetwork instance with 784 input values, 264 hidden neurons, and 10 output values
        nn = NaiveSideNet(784, 10, 264, 2, activation_function="relu", learning_rate=0.1, device=device)
        print("nn made")
        
        # train our network
        nn.train(images, labels, epochs=1)
        print("full train done")

        # test our trained network
        output = nn.deterministic_choose(test_images)
        
        # Compare outputs (with some tolerance for floating point differences)
        correct = (torch.argmax(output, dim=1) == torch.argmax(test_labels, dim=1))
        accuracy = correct.sum().item() / len(test_labels)
        
        if accuracy < 1.0:
            failures += 1
            print(f"We failed. failures = {failures}, accuracy = {accuracy}")

    end_time = time.perf_counter()
    print("accuracy:", (tries-failures)/tries, "\naverage time (seconds): ", (end_time - start_time)/tries)

'''# Initialize MNIST data loader
mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/MNIST/raw')

# Load the training data
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Get a random index
index = random.randrange(0, len(images)) 

# Display the image
print(mndata.display(images[index])) '''