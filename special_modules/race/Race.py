from Hash import *
import pdb

class Race():
  def __init__(self, params):
    self.range = params["range"]
    self.repetitions = params["repetitions"]
    self.power = params["power"]
    self.num_classes = params["num_classes"]
    self.rehash = params["rehash"]
    self.max_coord = params["max_coord"]
    self.min_coord = params["min_coord"]
    self.hashfunction = HashFunction.get(params["lsh_function"], num_hashes=self.power * self.repetitions)
    self.sketch_memory = np.zeros(shape=(self.num_classes, self.repetitions, self.range))
    self.max_range_hf = self.hashfunction.get_max()
    self.min_range_hf = self.hashfunction.get_min()
    self.range_hf = self.max_range_hf - self.min_range_hf + 1 # number of values
    self.params = params

    self.offset_arr = np.power(self.range_hf, np.arange(self.power))

    if not self.rehash: # rehashing is not allowed. so the actual value shoud be within range
        assert((self.range_hf)**self.power <= self.range)

  def decode(self, bucket):
    hashvalues = np.zeros(self.power)
    for i in range(len(hashvalues)):
        idx = len(hashvalues) - i - 1
        hashvalues[idx] = bucket // self.offset_arr[idx]  
        bucket = bucket % self.offset_arr[idx]
    hashvalues = hashvalues + self.min_range_hf
    return hashvalues
    
  
  def encode(self, hashvalues):
    return np.sum(np.multiply(self.offset_arr, hashvalues))

  def sketch(self, x, y):
    '''
      x : b x d 
      y : b x 1 \in [0,num_classes)
    '''
    hash_locations = self.hashfunction.compute(x) # b x (power*repetitions)
    hash_locations = hash_locations - self.min_range_hf
    offsets = np.tile(self.offset_arr, (x.shape[0], self.repetitions))
    assert(offsets.shape == hash_locations.shape)
    hash_location_offset_ed =  np.multiply(hash_locations, offsets)
    
    # TODO do this in parallel use torch and scatter
    for rep in range(self.repetitions):
      update_loc  = np.sum(hash_location_offset_ed[:,rep*self.power:(rep+1)*self.power], axis=1).astype(np.int32)
      if self.rehash:
          raise NotImplementedError
      for i in range(x.shape[0]):
        c = y[i]
        loc = update_loc[i]
        self.sketch_memory[c][rep][loc] += 1

  def get_dictionary(self):
    race_sketch = {}
    race_sketch["memory"] = self.sketch_memory
    race_sketch["hashfunction"] = self.hashfunction.get_dictionary()
    race_sketch["params"] = self.params
    return race_sketch

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
    hash_locations = self.hashfunction.compute(x) # b x (power*repetitions)
    hash_locations = hash_locations - self.min_range_hf
    offsets = np.tile(self.offset_arr, (x.shape[0], self.repetitions))
    assert(offsets.shape == hash_locations.shape)
    hash_location_offset_ed =  np.multiply(hash_locations, offsets)
    
    # TODO do this in parallel use torch and scatter
    kde_estimates = np.zeros(x.shape[0])
    for rep in range(self.repetitions):
      locs  = np.sum(hash_location_offset_ed[:,rep*self.power:(rep+1)*self.power], axis=1).astype(np.int32)
      if self.rehash:
          raise NotImplementedError
      for i in range(x.shape[0]):
        c = y[i]
        loc = locs[i]
        kde_estimates[i] += self.sketch_memory[c][rep][loc] / np.sum(self.sketch_memory[c][rep])
    kde_estimates /= self.repetitions
    return kde_estimates
      

