import numpy as np
import torch
from torch.utils import data
import pdb
from BasicData import BasicData
from special_modules.race.RaceGen import RaceGen
from tqdm import tqdm
import pickle

class RunningRaceParser(data.Dataset):
    def __init__(self, X_file, params):
        self.underlying_data = BasicData(params["underlying_data"])
        # some sanity checks
        assert(params["underlying_data"]["sampler"] == "simple")

        self.race = RaceGen(params["race"])
        self.save_final_sketch = None
        if "save_final_sketch" in params:
            self.save_final_sketch = params["save_final_sketch"]
        self.skip_rows = params["skip_rows"]
        self.Data = None
        self.Labels  = None
        self.pre_sketch_race()

        if self.save_final_sketch is not None:
            dictionary = self.race.get_dictionary()
            with open(self.save_final_sketch, "wb") as f:
                pickle.dump(dictionary, f)

    def pre_sketch_race(self):
        print("Pre processing the data to generate running race data")
        data = []
        labels = []
        self.underlying_data.reset()
        num_samples = self.underlying_data.len()
        batch_size = self.underlying_data.batch_size()
        num_batches = int(np.ceil(num_samples/batch_size))
        for i in tqdm(range(num_batches)):
            if self.underlying_data.end():
                break
            x, y = self.underlying_data.next()
            new_x = self.race.query_values(x) # query before sketching
            self.race.onDecay(x.shape[0])
            self.race.sketch(x, y)
            data.append(new_x)
            labels.append(y)

        self.Data = torch.cat(data)
        self.Labels = torch.cat(labels)
        self.Data = self.Data[self.skip_rows:, :]
        self.Labels = self.Labels[self.skip_rows:]


    def __len__(self):
        return self.Data.shape[0]

    def __getitem__(self, index):
        label = np.float(self.Labels[index].cpu())
        data_point = np.array(self.Data[index, :].cpu())
        return data_point, label

