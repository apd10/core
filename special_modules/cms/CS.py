''' This is general CountSketch. Implemetns both CS and CMS. 
    Also, the key can be multiple values. 
    Always uses 2-universal hash functions. We can increase it to p-universal. But then for num_keys > 1, it will be too much computation . But can be added in future
    parameters : 
      - topk 
      - range R
      - rep K
      - support to build this on GPU / CPU
      - num_keys

    Notes
      1. if you want a reversible loc to keys mapping. it is advised that you combine the keys yourself and then pass it
'''

import numpy as np
import array
#from random_number_generator import ConsistentRandomNumberGenerator as CRNG
from special_modules.crng.random_number_generator import ConsistentRandomNumberGenerator as CRNG
import pdb
import torch
import pandas as pd

def matmul_int(A, B, M1, M2):
    ''' int multiply A and B 
        int is not supported in torch
        M1: mod1 
        M2:
        A: batch x num_keys+1
        B : num_keys + 1 x K 
        returned is % M1 %M2

        returns : A x B  : batch x K 
    '''
    Bt = torch.transpose(B, 0, 1)
    Rs = []
    for i in range(Bt.shape[0]):
        b = Bt[i, :]
        r = torch.mul(A, b) % M1  # broadcast b over A  # %M1 %M2 is not distrubtive . ONlny %M1 is 
        r = torch.sum(r, dim=1) % M1 # B x 1
        Rs.append(r.reshape(-1,1))
    R = torch.cat(Rs, dim=1) %M1%M2
    return R
      

class TopKDs() :
    ''' This is a data structure for top k elements 
        To optimize find, insert, delete minimum and update operations
        we implement the datastructure by maintaining a hashmap (dictionary) 
        and heap. Note that O(log(n)) time is not possible by maintaining only 
        one of these
    '''
    def __init__(self, k):
        self.capacity = k;
        self.topk_df = None

    def show(self):
        print(self.topk_df)

    def insert(self, ids , values):
        # maybe a pandas dataframe might be faster
        # experimental . 
        ids = np.array(ids.cpu()) # num x num_keys
        values = np.array(values.cpu())
        new_df  = pd.DataFrame(ids)
        key_columns = ["C"+str(i) for i in range(ids.shape[1])] 
        new_df.columns = key_columns
        new_df.loc[:, "value"] =values

        if self.topk_df is None:
            self.topk_df = new_df
        else:
            # remove duplicates
            self.topk_df = pd.concat([self.topk_df, new_df], ignore_index=True)
            self.topk_df.drop_duplicates(subset=key_columns, keep='last', inplace=True)
            self.topk_df.sort_values('value', ascending=False, inplace=True)
            self.topk_df = self.topk_df.head(self.capacity)
        
    def getTop(self):
        return self.topk_df
   
class CountSketch() : 
    def __init__(self, seed, params):
        ''' d: number of hash functions
            R: range of the hash function. i.e. the memory
                to be used for count sketch
        '''
        self.params = params
        self.K = params["rep"]
        self.R = params["range"]
        self.num_keys = params["num_keys"]
        self.device_id = params["device_id"]
        self.topK = None
        self.random_seed = seed

        self.sketch_type = params["sketch_type"]
        self.recovery = params["recovery"]
        assert(self.recovery in ["min", "mean", "median"])

        self.is_cms = self.sketch_type == "CMS"
        self.is_cs = self.sketch_type == "CS"
        assert(self.is_cms or self.is_cs)

        if "topK" in params:
            self.topK = params["topK"]
        self.ignore_1 = False
        if "ignore_1" in params:
            self.ignore_1 = True
        self.is_decay_on = False
        if "decay" in params:
            self.is_decay_on = True
            self.decay_params = params["decay"]
            self.on_sample_alpha = None
            if "alpha" in self.decay_params:
                self.on_sample_alpha = self.decay_params["alpha"]
            if "half_life" in self.decay_params:
                hl = self.decay_params["half_life"]
                self.on_sample_alpha = np.power(2, -1/hl)
            assert(self.on_sample_alpha is not None)

        self.rng = CRNG(self.random_seed)
        self.sketch_memory = torch.zeros((self.K, self.R))
        random_numbers = np.array(self.rng.generate((self.num_keys + 1)*self.K * 2))  # 2 for H and G


        # (numkeys + 1) x K # add 1 to the data before applying this matrix
        self.HMatrix  = torch.LongTensor(random_numbers[0:(self.num_keys + 1) * self.K].reshape(self.num_keys + 1, self.K))
        self.GMatrix = None
        if self.is_cs:
            self.GMatrix  = torch.LongTensor(random_numbers[(self.num_keys + 1) * self.K: (self.num_keys + 1) * self.K * 2].reshape(self.num_keys + 1, self.K))
            assert (self.recovery != "min")
        
        if self.device_id != -1: # to put on device or not
            self.HMatrix = self.HMatrix.cuda(self.device_id)
            if self.GMatrix is not None:
                self.GMatrix = self.GMatrix.cuda(self.device_id)
            self.sketch_memory = self.sketch_memory.cuda(self.device_id)
        
        self.topkds = None
        if self.topK is not None:
            self.topkds = TopKDs(self.topK)

        

    def hash_locs (self, batch_keys):
        B = batch_keys.shape[0]
        ones_for_hash = torch.ones((B, 1), dtype=batch_keys.dtype)
        if self.device_id != -1:
            ones_for_hash = ones_for_hash.cuda(self.device_id)
        batch_keys = torch.cat((batch_keys, ones_for_hash), dim=1)
        hlocs = matmul_int(batch_keys, self.HMatrix, self.rng.big_prime, self.R) # B x K matmul_int gives: : (A X B) % M1 % M2
        gs = torch.ones((B, self.K))
        if self.device_id != -1:
            gs = gs.cuda(self.device_id)
        if self.GMatrix is not None:
            gs = torch.sign(matmul_int(batch_keys, self.GMatrix, self.rng.big_prime, 2) - 0.5) # B x K
        return hlocs, gs
        

    def insert(self, batch_keys, values): 
        # batch_keys = B x num_keys
        # values = B x 1
        B = batch_keys.shape[0]
        hlocs, gs = self.hash_locs(batch_keys)

        signed_values = torch.mul(gs, values) # B x K

        for i in range(self.K):
          self.sketch_memory[i].scatter_add_(0, hlocs[:,i], signed_values[:,i])

        # Insert the top K into the heap structure. Heap with CS setting is a bit unreliable. nonetheless i.i.d samples should work
        if self.topkds is not None:
          insert_batch = torch.unique(batch_keys, dim=0)
          nvalues = self.query(insert_batch) # 1 x B
          if insert_batch.shape[0] > self.topK:
              idx = torch.topk(nvalues, k=self.topK)[1] # topK should be small like 1000
              nvalues = nvalues[idx]
              insert_batch = insert_batch[idx, :] # min(topK,B) x num_keys
          self.topkds.insert(insert_batch, nvalues) 

    def query(self, batch_keys):  # returns 1 X B
        hlocs, gs = self.hash_locs(batch_keys)
        vs = []
        for i in range(self.K):
          v = torch.mul(self.sketch_memory[i][hlocs[:,i:(i+1)]], gs[:,i:(i+1)]) # v : B x 1
          vs.append(v)
        V = torch.cat(vs, dim=1)
        if self.recovery == "mean":
            return torch.mean(V, dim=1)
        elif self.recovery == "min":
            return torch.min(V, dim=1)[0]
        elif self.recovery == "median":
            V = torch.sort(V, dim=1)[0] # B x K
            if self.K %2 == 1:
                return V[:, self.K//2]
            else:
                return ((V[:, self.K//2 - 1] + V[:, self.K//2])/2)

    def get_top(self):
        self.topkds.topk_df.reset_index(drop=True, inplace=True)
        X =  self.topkds.topk_df
        if self.ignore_1:
            assert(self.is_cms)
            X = X[X.value > 1]
        return X

    def get_dictionary(self):
        dic = {}
        dic["params"] = self.params
        dic["random_seed"] = self.random_seed
        dic["sketch_memory"] = self.sketch_memory.cpu()
        if self.topkds is not None:
            dic["topK"] = self.topkds.topk_df
        return dic

    def set_dictionary(self, dic):
        self.sketch_memory = dic["sketch_memory"]
        self.topkds.topk_df = dic["topK"]
        if self.device_id != -1:
            self.sketch_memory = self.sketch_memory.cuda(self.device_id)
        assert(self.random_seed == dic["random_seed"])
        assert(self.params == dic["params"])

    def onDecay(self, samples=1):
        # add samples worth of decay
        if not self.is_decay_on:
            return
        decay = float(np.power(self.on_sample_alpha, samples))
        self.sketch_memory = self.sketch_memory * decay
        

if __name__ == "__main__":
    torch.manual_seed(101)
    X = torch.randint(low=0, high=10000, size=(100, 10)) 
    values = torch.rand(size=(100,1))
    params = {
        "rep" : 5,
        "range" : 1000,
        "num_keys" : 10,
        "device_id" : -1,
        "random_seed" : 101,
        "sketch_type" : "CMS",
        "recovery" : "min",
        "topK" : 20
    }
    cs = CountSketch(params)
    batch_size = 10
    for i in range(0, int(X.shape[0] // batch_size)):
        xs = X[i*batch_size:(i+1)*batch_size, :]
        if params["device_id"] !=-1:
            xs = xs.cuda(params["device_id"])
        vs = values[i*batch_size:(i+1)*batch_size, :]
        if params["device_id"] !=-1:
            vs = vs.cuda(params["device_id"])
        cs.insert(xs, vs)
    print(cs.topkds.topk_df)
