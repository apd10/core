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

class RaceSampler(data.Dataset):
    def __init__(self, pickle_file, params):
        super(RaceSampler, self).__init__()
        with open(pickle_file, "rb") as f:
          self.race_pickle = pickle.load(f)
        self.length = params["epoch_samples"]
        self.race = Race(self.race_pickle["params"])
        self.race.sketch_memory = self.race_pickle["memory"]
        self.race.hashfunction.set_dictionary(self.race_pickle["hashfunction"])
        # computing class probabilities
        class_counts = np.zeros(self.race.num_classes)
        for c in np.arange(self.race.num_classes):
          class_counts[c] = np.sum(self.race.sketch_memory[c][0])
        self.class_prob = class_counts / np.sum(class_counts)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = np.argmax(np.random.multinomial(1, self.class_prob, size=None))
        rep = np.random.randint(0, self.race.repetitions)
        sketch = self.race.sketch_memory[label][rep] # 1 x range  array of counts
        bucket = np.argmax(np.random.multinomial(1, sketch/np.sum(sketch), size=None))
        hash_values = self.race.decode(bucket)
        Weq,Beq = self.race.get_equations(hash_values, rep, len(hash_values))
        polytope = Polytope(A=Weq, b=Beq)
        
        # getting a point inside the polytope
        x_random = np.zeros(Weq.shape[1]) # if its SRP then 0 is already in the polygon!
        minover = MinOver(polytope=polytope)
        point, convergence = minover.run(starting_point=x_random, max_iters=10, speed=0.01)

        assert(convergence)
        assert(polytope.check_inside(point))
        hitandrun = HitAndRun(polytope=polytope, starting_point=point)
        sample = hitandrun.get_samples(n_samples=1, thin=100)
        kde_estimate = self.race.query(sample, [label])
        print(index, kde_estimate)
        return sample[0],label
