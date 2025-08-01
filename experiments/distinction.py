import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Define the Distinction model

# we are building an evolutionary model 
# the model will use NEAT at a high level 
# each generation will have a population of 1000 individuals 
# each individual will be an agent that will have initial weights and biases 
# these weights and biases will be their genotype that gets passed down or mutated
# the amount of mutation itself will be a gene 
# Q: what dataset will we use? 
# A: we will use the MNIST dataset first then we will use the CIFAR-10 dataset
class Distinction(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super(Distinction, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train(self, train_loader, epochs, learning_rate): 
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs): 
            for i, (images, labels) in enumerate(train_loader): 
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()