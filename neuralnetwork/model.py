import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn

def to_np(y):
    '''Converts torch tensor to numpy array
    params:
        y: torch tensor
    returns:
        y: numpy array
    '''
    return y.cpu().detach().numpy()

def cu_fraction(Sn):
    '''Calculates the fraction of Cu in the catalyst
    params:
        Sn: float, the percentage of Sn in the catalyst
    returns:
        Cu: float, the percentage of Cu in the catalyst
    '''
    if Sn <= 1:
        Cu = 1 - Sn
    else:
        raise ValueError('Sn percent must be less than or equal to 1')
    return Cu

def get_weight(Sn):
    '''Calculates the weight of the catalyst
    params:
        Sn: float, the percentage of Sn in the catalyst
    returns:
        weight: float, the weight of the catalyst
    '''
    # create the structure
    if Sn <= 1:
        weight = (1 - Sn)*63.546 + (Sn)*118.71
    else:
        raise ValueError('Sn percent must be less than or equal to 1')
    return weight

class MLP(nn.Module):
    """
    A simple multilayer perceptron
    params:
        layers: list of ints, the number of nodes in each layer
    returns:
        out: tensor, the output of the neural network
    """
    def __init__(self, layers):
        super().__init__() 
        self.layers = layers
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        for i in range(len(self.layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0) # set weights to xavier normal
            nn.init.zeros_(self.linears[i].bias.data) # set biases to zero

    def forward(self, x):
        if torch.is_tensor(x) is not True:         
            x = torch.from_numpy(x)  
        
        act_hid_out = x.float() # input layer

        for i in range(len(self.layers)-2): # input -- penultimate layer. apply activation to all but the last layer
            hid_out = self.linears[i](act_hid_out)    
            act_hid_out = self.activation(hid_out)
        
        out = nn.functional.softplus(self.linears[-1](act_hid_out)) # ensure output is positive
        return out
    

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    params:
        tolerance: int, number of epochs to wait before stopping training
        min_delta: float, minimum difference between new loss and old loss to be considered as an improvement
    returns:
        None
    """
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True