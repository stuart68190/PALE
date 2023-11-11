import torch.nn as nn

class BiggerNet(nn.Module):
    """
    A neural network with three fully connected layers.
    The input size is 784, output size is 10, and the hidden layer sizes are 500 and 300.
    Uses ReLU activation function for the hidden layers.
    """
    def __init__(self):
        super(BiggerNet, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SimpleNet(nn.Module):
    """
    A simple neural network with two fully connected layers.
    The input size is 784, output size is 10, and the hidden layer size is 5.
    Uses ReLU activation function for the hidden layer.
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 5)
        self.fc2 = nn.Linear(5, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x