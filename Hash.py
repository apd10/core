import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from hashes.list_hashfunctions import *

class HashFunction:
  def get(params, num_hashes):
    hash_func = None
    if params["name"] == "l2lsh":
      hash_func = L2LSH(params["l2lsh"], num_hashes)
    elif params["name"] == "srp":
      hash_func = SRP(params["srp"], num_hashes)
    else:
      raise NotImplementedError
    return hash_func
