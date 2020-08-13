import numpy as np
import torch
from torch.utils import data
import pandas as pd
import pdb
import pickle
from special_modules.race.Race import *
from special_modules.sampling_polytopes.hitandrun.hitandrun.polytope import *
from special_modules.sampling_polytopes.hitandrun.hitandrun.hitandrun import *
from special_modules.sampling_polytopes.hitandrun.hitandrun.minover import *
import concurrent
from tqdm import tqdm
from scipy.sparse import csr_matrix

global_vars = {
  "race" : None,
  "class_prob" : None,
  "max_iters" : None,
  "speed" : None
}


def sample_m1(rep, label, bucket):
    race = global_vars["race"]
    class_prob = global_vars["class_prob"]
    hash_values = race.decode(bucket)
    Weq,Beq = race.get_equations(hash_values, rep, len(hash_values))
    polytope = Polytope(A=Weq, b=Beq)
    
    # getting a point inside the polytope
    x_random = np.zeros(Weq.shape[1]) # if its SRP then 0 is already in the polygon!
    minover = MinOver(polytope=polytope)
    for i in range(10):
      point, convergence = minover.run(starting_point=x_random, max_iters=global_vars["max_iters"], speed=global_vars["speed"])
      if convergence:
        break;

    if not convergence:
        return None
    assert(polytope.check_inside(point))
    hitandrun = HitAndRun(polytope=polytope, starting_point=point)
    sample = hitandrun.get_samples(n_samples=1, thin=100)
    return sample[0],label

def sample_batch(num, kde_lb):
    race = global_vars["race"]
    class_counts = global_vars["class_prob"] * num
    class_counts = class_counts.astype(int) + 1
    class_prob = global_vars["class_prob"]
    num_samples = 0
    Datas = []
    Labels = []

    while np.sum(class_counts) > 0 and num_samples < num:
        print("Samples", num_samples, "/", num)
        print(class_counts)
        print(class_prob)
        data = []
        labels = []
        with concurrent.futures.ProcessPoolExecutor(40) as executor:
            futures = []
            print("submitting jobs")
            this_num = min((num-num_samples)*10, num)
    
            for i in tqdm(range(this_num)):
                label = np.argmax(np.random.multinomial(1, class_prob, size=None))
                rep = np.random.randint(0, race.repetitions)
                sketch = race.sketch_memory[label] # 1 x range  array of counts
    
                sketch_data = sketch.data[sketch.indptr[rep]:sketch.indptr[rep+1]]
                sketch_cols = sketch.indices[sketch.indptr[rep]:sketch.indptr[rep+1]]
                bucket_idx = np.argmax(np.random.multinomial(1, sketch_data/np.sum(sketch_data), size=None))
                bucket = sketch_cols[bucket_idx]
                futures.append(executor.submit(sample_m1, rep, label, bucket))
            print("waiting for executions")
            for res in tqdm(concurrent.futures.as_completed(futures), total=this_num):
              d = res.result()
              if d is not None:
                  data.append(d[0])
                  labels.append(d[1])
        data =  np.array(data)
        labels = np.array(labels)
        if kde_lb is not None:
          kde_estimates = race.query(data, labels)
          data = data[kde_estimates > kde_lb]
          labels = labels[kde_estimates > kde_lb]
        num_samples += len(labels)
        # update class prob
        ls, cts = np.unique(labels, return_counts=True)
        for li in range(len(ls)):
          class_counts[ls[li]] = max(0, class_counts[ls[li]] - cts[li])
        class_prob = class_counts / np.sum(class_counts)
            
        Datas.append(data)
        Labels.append(labels)
    Data = np.concatenate(Datas, axis=0)
    Label = np.concatenate(Labels, axis=0)
    return Data, Label
        


class RaceSamplerPreProc(data.Dataset):
    def __init__(self, pickle_file, params):
        global global_vars
        super(RaceSamplerPreProc, self).__init__()
        with open(pickle_file, "rb") as f:
          self.race_pickle = pickle.load(f)
        self.length = params["epoch_samples"]
        self.parallel_batch = params["parallel_batch"]
        self.method = params["method"]
        self.kde_lb = None
        if "kde_lb" in params:
            self.kde_lb = params["kde_lb"]
        self.race = Race(self.race_pickle["params"])
        self.race.sketch_memory = self.race_pickle["memory"]
        self.race.hashfunction.set_dictionary(self.race_pickle["hashfunction"])
        self.race.class_counts = self.race_pickle["class_counts"]
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
            data,labels = sample_batch(self.parallel_batch, self.kde_lb)
            if self.Data is None:
                self.Data = data
                self.labels = labels
            else:
                self.Data = np.concatenate([self.Data, data], axis=0)
                self.labels = np.concatenate([self.labels, labels], axis=0)
              
            
        return self.Data[index],self.labels[index]
        

        

