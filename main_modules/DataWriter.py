import torch
from torch import nn
from torch.autograd import Variable
from Data import *
from Model import *
from Optimizer import *
from Loss import *
from ProgressEvaluator import *
import pdb

class DataWriter:
    ''' class to generate and write data '''
    def __init__(self, params):
        self.epochs = params["epochs"]
        # data
        self.train_data = Data(params["train_data"])
        self.write_data_file = params["write_data_file"]

    def loop(self):
        epoch = 0
        with open(self.write_data_file, "wb") as f:
          while epoch < self.epochs :
              self.train_data.reset()
              while not self.train_data.end():
                  x, y = self.train_data.next()
                  y = y.reshape(len(y),1).float()
                  data = torch.cat([y, x], axis=1)
                  np.savetxt(f, data, fmt="%.6f")
              epoch = epoch + 1
