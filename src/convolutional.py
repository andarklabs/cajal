# Building a Convolutional Neural Network
# we wish to make it so that is can be used 
# both efficiently and flexibly as part of the cajal ecosystem

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
import torch.nn.functional as F

#import mnist data from torchvision.datasets
import torchvision.datasets as datasets


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

class ConvolutionalNeuralNetwork(nn.Module):    
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.epochs = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    

    def train(self, train_loader, optimizer = optim.Adam, criterion = ""):
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()    


    def test(self, test_loader):
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)    


if __name__ == "__main__":
    model = ConvolutionalNeuralNetwork()
    trainloader = load_data()
    model.train(train_loader=trainloader)



