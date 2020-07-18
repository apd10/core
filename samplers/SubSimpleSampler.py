import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pdb
class SubSimpleSampler:
    def __init__(self, dataset, params):
      self.dataset = dataset
      self.batch_size = params["batch_size"]
      self.frac = params["frac"]
      self.total_size = int(dataset.__len__() * self.frac)
      temp = np.arange(dataset.__len__())
      np.random.shuffle(temp)
      self.indices = temp[:self.total_size]
      self.current_idx = 0

    def reset(self):
      self.current_idx = 0
    def next(self):
      xs = []
      ys = []
      for i in range(self.current_idx, min(self.current_idx + self.batch_size, self.total_size)):
          x,y = self.dataset[self.indices[i]]
          xs.append(x)
          ys.append(y)
      X = np.stack(xs)
      y = np.stack(ys)
      self.current_idx = self.current_idx + self.batch_size
      return X.shape[0], (torch.FloatTensor(X), torch.LongTensor(y))
