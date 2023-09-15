import torch
import torch.nn as nn
from network import BasicNeuralNetwork

def train(model: BasicNeuralNetwork):
    x = []
    y = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x.append([i, j, k])
                y.append(i + j + k)
    
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())
    
    for epoch in range(10000):
        optim.zero_grad()
        
        out = model(x)
        loss = loss_fn(out.squeeze(), y)
        
        if (epoch % 1000 == 0):
            print(f"the loss is currently: {loss.item()}")
        
        loss.backward()
        optim.step()

    return model