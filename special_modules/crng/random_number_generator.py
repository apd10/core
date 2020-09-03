from sklearn.utils import murmurhash3_32 as mmh
import numpy as np
import pdb
from sklearn.utils import murmurhash3_32

def Hfunction(m, seed=None):
    #if seed is None:
    #    seed = np.random.randint(0,100000)
    return lambda x : murmurhash3_32(key=x, seed=seed, positive=True) % m


class ConsistentRandomNumberGenerator():
    def __init__(self,seed):
      self.seed = seed
      self.big_prime = 1000_000_0139_9
      self.hfunc = Hfunction(self.big_prime, seed)
    
    def generate(self, k):
      ''' give k random numbers '''
      idxs = np.arange(0,k)
      idxs = np.square(idxs + 2)
      return [ self.hfunc(int(idx)) for idx in idxs]


if __name__ == '__main__':
    crng = ConsistentRandomNumberGenerator(10100)
    for i in range(0,10):
       print(crng.generate(i))
