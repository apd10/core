import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, dimension, num_layers, hidden_size, num_class=2):
        super(FCN, self).__init__()

        self.first_layer = nn.Linear(dimension, hidden_size)
        self.first_layer_relu = nn.ReLU()
        mid_layers = []
        for i in range(num_layers - 2):
            mid_layers.append(nn.Linear(hidden_size, hidden_size))
            mid_layers.append(nn.ReLU())
        self.mid_layers = nn.Sequential(*mid_layers)

        self.last_layer = nn.Linear(hidden_size, num_class)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.first_layer_relu(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.last_layer(x)
        x = F.log_softmax(x, dim=1)
        return x


