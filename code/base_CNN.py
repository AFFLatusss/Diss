import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):
    
    def __init__(self):
        """Initialise a base network"""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        
        self.fc1 = nn.Linear(16*9*9, 120)  #Dimension after convolution and pooling 
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 4)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = x.view(-1, 16*9*9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

