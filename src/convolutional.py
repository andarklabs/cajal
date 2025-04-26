from dense import NeuralNetwork



class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, initial_learning_rate=0.1):
        super().__init__(layers, initial_learning_rate)

    def forward(self, X):
        return super().forward(X)

    def backward(self, X, y):
        return super().backward(X, y)
