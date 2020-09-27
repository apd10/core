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


def sample_m1(rep, label, hash_values, num_points):
    race = global_vars["race"]
    class_prob = global_vars["class_prob"]
    # Gonna write W.x  target_values * width -B . W : D x D  B : D x 1
    ds = []
    labels = []

    for i in range(num_points):
        target_values = np.random.uniform(size=len(hash_values)) + hash_values 
        W = race.get_w(rep, len(hash_values)) # W : P X D
        B = race.get_b(rep, len(hash_values)) # B : P
        width = race.get_r()
    
        assert(W.shape[1] >=  W.shape[0])
        # extra equations if required
        if W.shape[1] > W.shape[0]:
            print("TEST this code")
            pdb.set_trace() # test this code
            p = W.shape[0]
            dim = W.shape[1]
            extra = dim - p
            coords = np.random.randint(0, dim, size=extra)
            coord_values = np.random.uniform(size=extra)
            coord_values = coord_values * (race.max_coord - race.min_coord) + race.min_coord
            Wextra = np.identity(dim)[coords, :]
            Bextra = coord_values
            W = np.concatenate([W, Wextra], axis=0)
            B = np.concatenate([B, Bextra], axis=0)
    
        point = np.linalg.solve(W, width*target_values - B)
        ds.append(point)
        labels.append(label)
        #Weq,Beq = race.get_hf_equations(np.array(hash_values), rep, len(hash_values))
        #polytope = Polytope(A=Weq, b=Beq)
        #assert(polytope.check_inside(point))
    D = np.stack(ds, axis=0)
    L = np.array(labels)
    return D, L

def sample_classrep(rep, label, num):
    race = global_vars["race"]
    sketch = race.sketch_memory[label][rep] # Ace
    topk_df = sketch.get_top_buckets()
    values = topk_df["value"].values
    hashcols = [a for a in topk_df.columns if a.startswith("C")]
    bucket_counts = values/np.sum(values) * num
    ds = []
    labels = []
    for bucket_idx in range(len(bucket_counts)):
        count = np.floor(bucket_counts[bucket_idx])
        frac = bucket_counts[bucket_idx]  - count
        count = count + np.random.binomial(1, frac)
        count = int(count)
        if count == 0:
            continue
        hash_values = topk_df[hashcols].loc[bucket_idx, :].values
        d = sample_m1(rep, label, hash_values, count) # d = ([x1, x2, ..], [l1, l2, ..])
        ds.append(d[0])
        labels.append(d[1])

    # randomize
    D = np.concatenate(ds, axis=0)
    L = np.concatenate(labels)

    idx = np.arange(D.shape[0])
    np.random.shuffle(idx)
    D = D[idx]
    L = L[idx] # useless
    return D, L

def sample_batch(num):
    race = global_vars["race"]
    norm_info = False
    if "store_norms" in race.params:
        norm_info = race.params["store_norms"]
    class_counts = global_vars["class_prob"] * num
    class_counts = class_counts.astype(int) + 1
    num_samples = 0
    ds = []
    labels = []

    pll_num = 0
    with concurrent.futures.ProcessPoolExecutor(40) as executor:

        futures = []
        for this_class in range(len(class_counts)):
            this_class_count = class_counts[this_class]
            for rep in range(race.repetitions):
                this_class_rep_count = (this_class_count // race.repetitions) + 1
                #d = sample_classrep(rep, this_class, this_class_rep_count)
                #pdb.set_trace()
                futures.append(executor.submit(sample_classrep, rep, this_class, this_class_rep_count))
                pll_num += 1
            
        print("waiting for executions")

        for res in tqdm(concurrent.futures.as_completed(futures), total=pll_num):
              d = res.result()
              ds.append(d[0])
              labels.append(d[1])

    # randomize order
    D = np.concatenate(ds, axis=0)
    L = np.concatenate(labels)

    idx = np.arange(D.shape[0])
    np.random.shuffle(idx)
    D = D[idx]
    L = L[idx] # useless
    return D, L
        


class RaceSamplePPD(data.Dataset):
    def __init__(self, pickle_file, params):
        global global_vars
        super(RaceSamplePPD, self).__init__()
        with open(pickle_file, "rb") as f:
          self.race_pickle = pickle.load(f)
        self.length = params["epoch_samples"]
        self.parallel_batch = params["parallel_batch"]
        self.method = params["method"]
        self.kde_lb = None
        if "kde_lb" in params:
            self.kde_lb = params["kde_lb"]
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
