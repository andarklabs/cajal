from dense import NeuralNetwork



class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, conv_layers, dense_layers, initial_learning_rate=0.1):
        super().__init__(dense_layers, initial_learning_rate)
        self.conv_layers = conv_layers

    def forward(self, X):
        return super().forward(X)

    def backward(self, X, y):
        return super().backward(X, y)
