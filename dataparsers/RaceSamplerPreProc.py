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

global_vars = {
  "race" : None,
  "class_prob" : None,
  "max_iters" : None,
  "speed" : None
}


def sample_m1(rep, label, bucket):
    race = global_vars["race"]
    class_prob = global_vars["class_prob"]
    sketch = race.sketch_memory[label][rep] # 1 x range  array of counts
    hash_values = race.decode(bucket)
    Weq,Beq = race.get_equations(hash_values, rep, len(hash_values))
    polytope = Polytope(A=Weq, b=Beq)
    
    # getting a point inside the polytope
    x_random = np.zeros(Weq.shape[1]) # if its SRP then 0 is already in the polygon!
    minover = MinOver(polytope=polytope)
    point, convergence = minover.run(starting_point=x_random, max_iters=global_vars["max_iters"], speed=global_vars["speed"])

    assert(convergence)
    if not convergence:
        return None
    assert(polytope.check_inside(point))
    hitandrun = HitAndRun(polytope=polytope, starting_point=point)
    sample = hitandrun.get_samples(n_samples=1, thin=100)
    return sample[0],label

def sample_batch(num):
    race = global_vars["race"]
    class_prob = global_vars["class_prob"]
    data = []
    with concurrent.futures.ProcessPoolExecutor(25) as executor:
        futures = []
        print("submitting jobs")
        for i in tqdm(range(num)):
            label = np.argmax(np.random.multinomial(1, class_prob, size=None))
            rep = np.random.randint(0, race.repetitions)
            sketch = race.sketch_memory[label][rep] # 1 x range  array of counts
            bucket = np.argmax(np.random.multinomial(1, sketch/np.sum(sketch), size=None))
            futures.append(executor.submit(sample_m1, rep, label, bucket))
        print("waiting for executions")
        for res in tqdm(concurrent.futures.as_completed(futures), total=num) :
          d = res.result()
          data.append(d)
    return data


class RaceSamplerPreProc(data.Dataset):
    def __init__(self, pickle_file, params):
        global global_vars
        super(RaceSamplerPreProc, self).__init__()
        with open(pickle_file, "rb") as f:
          self.race_pickle = pickle.load(f)
        self.length = params["epoch_samples"]
        self.parallel_batch = params["parallel_batch"]
        self.method = params["method"]
        self.race = Race(self.race_pickle["params"])
        self.race.sketch_memory = self.race_pickle["memory"]
        self.race.hashfunction.set_dictionary(self.race_pickle["hashfunction"])
        # computing class probabilities
        class_counts = np.zeros(self.race.num_classes)
        for c in np.arange(self.race.num_classes):
          class_counts[c] = np.sum(self.race.sketch_memory[c][0])
        self.class_prob = class_counts / np.sum(class_counts)
        self.method_params = params[params["method"]]

        #set global vars
        global_vars["race"] = self.race
        global_vars["class_prob"] = self.class_prob
        global_vars["max_iters"] = self.method_params["minover"]["max_iters"]
        global_vars["speed"] = self.method_params["minover"]["speed"]

        self.Data = []
    
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index + 1 > len(self.Data):
            d = sample_batch(self.parallel_batch)
            self.Data = self.Data + d
            
        return self.Data[index]
        

        

