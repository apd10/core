import numpy as np
from scipy.stats import norm # for P_L2
import math 
from scipy.special import ndtr
import pdb

class SRP():
  def __init__(self, params, num_hashes): 
    # N = number of hashes 
    # d = dimensionality
    self.N = num_hashes
    self.d = params["dimension"]
    # set up the gaussian random projection vectors
    self.W = np.random.normal(size = (self.d, self.N)) # D x num_hashes


  def compute(self,x):  # b x d
    values =   np.matmul(x, self.W) >= 0
    values = values.astype(np.int32)
    if np.sum(values > self.get_max()) > 0:
        pdb.set_trace()
    return values
    

  def get_max(self):
    return 1

  def get_min(self):
    return 0

  def get_dictionary(self):
    dic = {}
    dic["N"] =self.N
    dic["d"] =self.d
    dic["W"] = self.W
    return dic
