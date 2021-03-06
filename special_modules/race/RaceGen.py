''' This is the general RACE with each ACE implemented as rehashed into CMS .
       We will keep enough flexibility to be able to use original RACE (with arrays) and with or without rehashing.

parameters :
  - L : number of hash tables for LSH . i.e. in context of RACE, number of repitions
  - ACE:
    - R : Range of Array
    - K : number of repetitions
    - rehash : assert(rehash => K=1)
    - topK :
'''

import pdb
from scipy.sparse import csr_matrix
import torch
import numpy as np # only for equation
from Hash import *

#from random_number_generator import ConsistentRandomNumberGenerator as CRNG
from special_modules.crng.random_number_generator import ConsistentRandomNumberGenerator as CRNG
from special_modules.race.Ace import *

class RaceGen:
  def __init__(self, params):
    self.repetitions = params["repetitions"]
    self.power = params["power"]
    self.num_classes = params["num_classes"]
    self.max_coord = params["max_coord"]
    self.min_coord = params["min_coord"]
    self.hashfunction = HashFunction.get(params["lsh_function"], num_hashes=self.power * self.repetitions)
    self.random_seed = params["random_seed"]
    if "np_seed" in params:
      self.np_seed = params["np_seed"]
      np.random.seed(self.np_seed)

    self.store_norms = False
    if "store_norms" in params:
        self.store_norms = params["store_norms"]

    
    self.params = params
    self.ace_type = params["ace_type"]
    self.ace_params = params["ace_params"]
    self.sketch_memory = {}
    self.crng = CRNG(self.random_seed)
    random_numbers = self.crng.generate(self.num_classes * self.repetitions)

    for i in range(self.num_classes):
        self.sketch_memory[i] = []
        for rep in range(self.repetitions):
            seed = random_numbers[i*self.repetitions + rep]
            self.sketch_memory[i].append(Ace.get(self.ace_type, seed, self.ace_params, self.store_norms))
    self.class_counts = np.zeros(self.num_classes)

  def sketch(self, x, y):
    '''
      x : b x d 
      y : b x 1 \in [0,num_classes)
    '''
    hashes = self.hashfunction.compute(x) # b x (power*repetitions)
    if self.store_norms:
        norms = torch.sqrt(torch.sum(torch.mul(x,x), axis=1))

    for rep in range(self.repetitions):
      hash_values  = hashes[:,rep*self.power:(rep+1)*self.power]
      for c in np.arange(self.num_classes):
          examples_perclass = hash_values[y == c]
          if rep ==0: # add only once
              self.class_counts[c] += examples_perclass.shape[0]
          if examples_perclass.shape[0] > 0:
              norms_perclass = None
              if self.store_norms:
                  norms_perclass = norms[y == c]
                  norms_perclass = norms_perclass.reshape(examples_perclass.shape[0],1)
              self.sketch_memory[c][rep].insert(examples_perclass, torch.ones((examples_perclass.shape[0], 1)), norms_perclass)

  def get_dictionary(self):
    race_sketch = {}

    race_sketch["params"] = self.params
    race_sketch["memory"] = {}
    for c in range(self.num_classes):
      race_sketch["memory"][c] = []
      for r in range(self.repetitions):
          race_sketch["memory"][c].append(self.sketch_memory[c][r].get_dictionary())
    race_sketch["hashfunction"] = self.hashfunction.get_dictionary()
    race_sketch["class_counts"] = self.class_counts
    return race_sketch

  def set_dictionary(self, dic):
    for c in range(self.num_classes):
        for r in range(self.repetitions):
            self.sketch_memory[c][r].set_dictionary(dic["memory"][c][r])

    self.hashfunction.set_dictionary(dic["hashfunction"])
    self.class_counts = dic["class_counts"]
    assert(self.params ==  dic["params"])

  def get_hf_equations(self, hash_values, rep, chunk_size):
    W_heq, b_heq = self.hashfunction.get_equations(hash_values, rep, chunk_size)
    return W_heq, b_heq

  def get_w(self, rep, chunk_size):
    return self.hashfunction.get_w(rep, chunk_size)

  def get_b(self, rep, chunk_size):
    return self.hashfunction.get_b(rep, chunk_size)

  def get_r(self):
    return self.hashfunction.get_r()

  def get_bounding_equations(self):
    W_max = np.identity(self.hashfunction.d)
    b_max = np.ones(self.hashfunction.d) * self.max_coord
    W_min = np.identity(self.hashfunction.d) * -1
    b_min = np.ones(self.hashfunction.d) * self.min_coord * -1
    W_total = np.concatenate([W_max, W_min])
    b_total = np.concatenate([b_max, b_min])
    return W_total, b_total

  def get_equations(self, hash_values, rep, chunk_size):
    # get hash equations
    W_heq, b_heq = self.hashfunction.get_equations(hash_values, rep, chunk_size)
    # get bounding boxes
    W_max = np.identity(self.hashfunction.d)
    b_max = np.ones(self.hashfunction.d) * self.max_coord
    W_min = np.identity(self.hashfunction.d) * -1
    b_min = np.ones(self.hashfunction.d) * self.min_coord * -1
    W_total = np.concatenate([W_heq, W_max, W_min])
    b_total = np.concatenate([b_heq, b_max, b_min])
    return W_total, b_total


  def query(self, x, y):
    ''' return the K.D.E value for x w.r.t class y '''
    raise NotImplementedError

  def query_values(self, x):
    ''' aggregate the counts of all aces we have for each class
        if y is not specified. '''
    ''' this is batch wise
    '''
    hashes = self.hashfunction.compute(x) # b x (power*repetitions)
    values = torch.zeros((x.shape[0], self.num_classes * self.repetitions))
    for c in range(self.num_classes):
        for rep in range(self.repetitions):
            hash_values  = hashes[:,rep*self.power:(rep+1)*self.power]
            idx = self.repetitions * c + rep
            ace = self.sketch_memory[c][rep]
            cts = ace.query(hash_values)
            values[:, idx] = cts
    return values

  def onDecay(self, samples):
    for c in range(self.num_classes):
      for rep in range(self.repetitions):
        ace = self.sketch_memory[c][rep]
        ace.onDecay(samples)
              

            
      

