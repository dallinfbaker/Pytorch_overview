# import torch

import torch.nn as nn

class BasicNeuralNetwork(nn.Module):
    def __init__(self):
        super(BasicNeuralNetwork, self).__init__()
        self.fc = nn.Linear(3, 3, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1, bias=True)
    
    def forward(self, x):
        output = self.fc(x)
        output = self.relu(output)
        output = self.fc2(output)
        return output