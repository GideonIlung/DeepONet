#IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

#custom libaries
from model import DeepONet

#training algorithm#
def train_DON(model:DeepONet,x_branch,x_trunk,y_,lr:float=0.001,batch_size=32,epochs=100,log=True):
    """
        trains a deep operator network

        Parameters:
            model    (DeepONet)     : the network to be trained
            x_branch (torch.tensor) : the branch input data
            x_trunk  (torch.tensor) : the trunk input data
            y        (torch.tensor) : the targets
    """

    #creating datahandler#
    dataset = Datahandler(x_branch,x_trunk,y_) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    #using standard MSE loss#
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #getting trunk data#
    trunk = convert_np_to_tensor(x_trunk)

    #creating training loop#
    for epoch in range(epochs):
        losses = []
        for branch,_,y in dataloader:
            
            #removing previous gradients#
            optimizer.zero_grad()

            #forward pass through model#
            output = model.forward(branch,trunk)
            loss = criterion(output,y)

            # Backward pass
            loss.backward()

            #calculate avg loss across batches#
            losses.append(loss.item())

            # Update parameters
            optimizer.step()

        avg_loss = np.mean(losses)

        if log == True:
            print('='*30)
            print(f"loss at epoch {epoch}:{avg_loss}") 
            print('='*30)

def convert_np_to_tensor(array):
        if isinstance(array, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array

class Datahandler(Dataset):
    """
        Handles the training Dataset 
        for the DeepONet
    """
    def __init__(self, x_branch_, x_trunk_, y_):

        x_branch = self.convert_np_to_tensor(x_branch_)
        x_trunk = self.convert_np_to_tensor(x_trunk_)
        y = self.convert_np_to_tensor(y_)

        self.x_batch = x_branch
        self.x_trunk = x_trunk
        self.y = y

    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array
    
    def __len__(self):
        return len(self.y)  # Assuming x_batch and x_trunk have the same length as y

    def __getitem__(self, index):
        return self.x_batch[index,:], self.x_trunk, self.y[index,:]