import numpy as np
from scipy.stats import norm # for P_L2
import math 
from scipy.special import ndtr
import pdb

class L2LSH():
  def __init__(self, params, num_hashes): 
    # N = number of hashes 
    # d = dimensionality
    # r = "bandwidth"
    self.N = num_hashes
    self.d = params["dimension"]
    self.r = params["bandwidth"]
    self.max_norm = None
    if "max_norm" in params:
        self.max_norm = params["max_norm"]

    # set up the gaussian random projection vectors
    self.W = np.random.normal(size = (self.d, self.N))
    # normalize
    # norms = np.sqrt(np.sum(np.multiply(self.W, self.W), axis=0)) #1 x N
    # self.W = self.W / norms
    # pdb.set_trace()
    self.b = np.random.uniform(low = 0,high = self.r,size = self.N)


  def compute(self,x):  # b x d
    values =   np.floor( (np.matmul(x, self.W) + self.b)/self.r ) # b x N
    if np.sum(values > self.get_max()) > 0:
        pdb.set_trace()
    return values
    

  def get_max(self):
    assert(self.max_norm is not None)
    return np.floor(self.max_norm / self.r)

  def get_min(self):
    assert(self.max_norm is not None)
    return np.floor(-self.max_norm / self.r)

  def get_dictionary(self):
    dic = {}
    dic["N"] =self.N
    dic["d"] =self.d
    dic["r"] =self.r
    dic["max_norm"] = self.max_norm
    dic["W"] = self.W
    dic["b"] = self.b
    return dic
