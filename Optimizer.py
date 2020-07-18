import torch
from torch import nn
class Optimizer:
  def get(model, params):
    if params["name"] == "adam":
      return  torch.optim.Adam(model.parameters(), lr=params["adam"]["lr"], weight_decay=params["adam"]["weight_decay"])
    else:
      raise NotImplementedError

    
