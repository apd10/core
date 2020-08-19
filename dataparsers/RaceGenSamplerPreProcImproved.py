import numpy as np
import torch
from torch.utils import data
import pandas as pd
import pdb
import pickle
from special_modules.race.RaceGen import *
from special_modules.sampling_polytopes.hitandrun.hitandrun.polytope import *
from special_modules.sampling_polytopes.hitandrun.hitandrun.hitandrun import *
from special_modules.sampling_polytopes.hitandrun.hitandrun.minover import *
import concurrent
from tqdm import tqdm
from scipy.sparse import csr_matrix
import scipy

global_vars = {
  "race" : None,
  "class_prob" : None,
  "max_iters" : None,
  "speed" : None
}


def sample_m1(rep, label, hash_values, n_samples):
    race = global_vars["race"]
    class_prob = global_vars["class_prob"]
    Weq,Beq = race.get_equations(np.array(hash_values), rep, len(hash_values))
    polytope = Polytope(A=Weq, b=Beq)
    
    res = scipy.optimize.linprog(c=np.zeros(Weq.shape[1]), A_ub = Weq, b_ub = Beq)
    point = res.x

    assert(polytope.check_inside(point))
    hitandrun = HitAndRun(polytope=polytope, starting_point=point)
    samples = hitandrun.get_samples(n_samples=n_samples, thin=100) # need better implementation for this. try vaidya-walk
    return samples, np.repeat(label, n_samples)

def sample_batch(total_num):
    race = global_vars["race"]
    class_counts = global_vars["class_prob"] * total_num
    class_counts = class_counts.astype(int) + 1
    class_prob = global_vars["class_prob"]
    num_samples = 0
    datas = []
    labels = []
    
    temp = 0
    with concurrent.futures.ProcessPoolExecutor(50) as executor:
      futures = []
      for rep in range(race.repetitions):
          for c in range(len(class_counts)):
              this_rep_num = class_counts[c] / race.repetitions
              print("Rep", rep, "Class", c)
              sketch = race.sketch_memory[c][rep]
              topk_df = sketch.get_top_buckets()
              values = topk_df["value"].values
              bucket_counts = values/np.sum(values) * this_rep_num
              for bkt in range(len(bucket_counts)):
                  hash_values = topk_df.loc[bkt, :].values[:-1]
                  fcount = 0
                  if bucket_counts[bkt] < 1:
                      bern = np.random.binomial(1, p=bucket_counts[bkt])
                      if bern == 1:
                          fcount = 1
                  else:
                      fcount = int(bucket_counts[bkt])
                  if fcount > 0:
                      #sample_m1(rep, c, hash_values, fcount)
                      futures.append(executor.submit(sample_m1, rep, c, hash_values, fcount))
                      temp = temp + 1
                  
      print("waiting for executions")
      for res in tqdm(concurrent.futures.as_completed(futures), total=temp):
          d = res.result()
          if d is not None:
              datas.append(d[0])
              labels.append(d[1])
    pdb.set_trace()
    Data = np.concatenate(datas, axis=0)
    Labels = np.concatenate(labels, axis=0)
    return Data, Labels
        


class RaceGenSamplerPreProc(data.Dataset):
    def __init__(self, pickle_file, params):
        global global_vars
        super(RaceGenSamplerPreProc, self).__init__()
        with open(pickle_file, "rb") as f:
          self.race_pickle = pickle.load(f)
        self.length = params["epoch_samples"]
        self.parallel_batch = params["parallel_batch"]
        self.method = params["method"]
        self.race = RaceGen(self.race_pickle["params"]) # do the change for Race as well
        self.race.set_dictionary(self.race_pickle)
        # computing class probabilities

        self.class_prob = self.race.class_counts / np.sum(self.race.class_counts)
        self.method_params = params[params["method"]]

        #set global vars
        global_vars["race"] = self.race
        global_vars["class_prob"] = self.class_prob
        global_vars["max_iters"] = self.method_params["minover"]["max_iters"]
        global_vars["speed"] = self.method_params["minover"]["speed"]

        self.Data = None
        self.labels = None
    
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.Data is None or index + 1 > self.Data.shape[0]:
            data,labels = sample_batch(self.parallel_batch)
            if self.Data is None:
                self.Data = data
                self.labels = labels
            else:
                self.Data = np.concatenate([self.Data, data], axis=0)
                self.labels = np.concatenate([self.labels, labels], axis=0)
              
            
        return self.Data[index],self.labels[index]
