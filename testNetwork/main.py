import torch
from network import BasicNeuralNetwork
from trainer import train

def main():
    
    model = BasicNeuralNetwork()
    model.load_state_dict(torch.load('tripleAddModel.pt'))
    model.eval()
    
    model = train(model)
    torch.save(model.state_dict(), 'tripleAddModel.pt')
    
    x = [35,55,750000]
    x = torch.Tensor(x)
    print(model(x))

main()
    