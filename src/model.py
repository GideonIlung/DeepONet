#IMPORTS#
import torch
import torch.nn as nn
import numpy as np

class DeepONet(nn.Module):
    """
        Implementation of the Deep Operator Network
    """

    def __init__(self,n_branch:int,width:int,depth:int,p:int,act,n_trunk:int=1):
        """
            Creates the DON using the following parameters

            Parameters:
            n_branch (int) : the input size of the branch network
            n_trunk  (int) : the input size of the trunk network
            depth    (int) : number of layers in each network 
            width.   (int) : number of nodes at each layer
            p        (int) : output dimension of network
            act            : the activation function to be used
        """
        super(DeepONet, self).__init__()

        #creating the branch network#
        self.branch_net = MLP(input_size=n_branch,hidden_size=width,num_classes=p,depth=depth,act=act)
        self.branch_net.float()

        #creating the trunk network#
        self.trunk_net = MLP(input_size=n_trunk,hidden_size=width,num_classes=p,depth=depth,act=act)
        self.trunk_net.float()
        
        self.bias = nn.Parameter(torch.ones((1,)),requires_grad=True)
    
    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array

    
    def forward(self,x_branch_,x_trunk_):
        """
            evaluates the operator

            x_branch : input_function
            x_trunk : point evaluating at

            returns a scalar
        """

        x_branch = self.convert_np_to_tensor(x_branch_)
        x_trunk = self.convert_np_to_tensor(x_trunk_)
        
        branch_out = self.branch_net.forward(x_branch)
        trunk_out = self.trunk_net.forward(x_trunk,final_act=True)

        output = branch_out @ trunk_out.t() + self.bias
        return output

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth,act):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        #the activation function#
        self.act = act 

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))
        
    def forward(self, x,final_act=False):
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)  # No activation after the last layer

        if final_act == False:
            return x
        else:
            return torch.relu(x)



