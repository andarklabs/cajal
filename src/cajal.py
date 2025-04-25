from dense import NeuralNetwork as NN
import numpy as np
import dense


class MLP():
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # DONE: We will adjust the following with our own weight initializations later 
        self.W1 = dense.init_weights(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = dense.init_weights(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def f    
    
if __name__ == "__main__":
    network0 = NN(input_dim=2, hidden_dim=20, output_dim=1, learning_rate=0.1) # we can build a control network here

