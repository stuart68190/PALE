import torch.nn as nn

#Defining the convolutional neural network
class ConvolvedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        padding = (2, 2, 2, 2)  # (padding_left, padding_right, padding_top, padding_bottom)
        x = nn.functional.pad(x, padding, "constant", 0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

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
    
class SimpleNet2(nn.Module):
    """
    A simple neural network with two fully connected layers.
    The input size is 784, output size is 10, and the hidden layer size is 512.
    Uses ReLU activation function for the hidden layer.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x