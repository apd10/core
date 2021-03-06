import torch
from torch import nn
from torch.autograd import Variable
from Data import *
import pdb
import numpy as np
import pickle
from special_modules.race.Race import *
from tqdm import tqdm


class RaceSketch:
    def __init__(self, params):
        self.data = Data(params["data"])
        self.pickle_file = params["save_sketch"]
        self.np_seed = params["np_seed"]
        self.race = Race(params["race"])
        np.random.seed(self.np_seed)
        torch.manual_seed(self.np_seed)

    def sketch(self):
        self.data.reset()
        num_samples = self.data.len()
        batch_size = self.data.batch_size()
        num_batches = int(np.ceil(num_samples/batch_size))
        for i in tqdm(range(num_batches)):
            if self.data.end():
              break
            x, y = self.data.next()
            x = np.array(x)
            y = np.array(y)
            self.race.sketch(x, y)
        sketch = self.race.get_dictionary()
        with open(self.pickle_file, "wb") as f:
            pickle.dump(sketch, f)
