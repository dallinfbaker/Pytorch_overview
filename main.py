import torch
from network import BasicNeuralNetwork
from trainer import train

def main():
    
    model = BasicNeuralNetwork()
    
    model = train(model)
    torch.save(model.state_dict(), 'tripleAddModel.pt')
    
    x = [35,55,750000]
    x = torch.Tensor(x)
    print(model(x))

    # You can load the model with this code 
    #   if you want to evaluate things with it
    # model.load_state_dict(torch.load('tripleAddModel.pt'))
    # model.eval()

main()
    