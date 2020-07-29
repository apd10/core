import numpy as np
import torch
from torch.utils import data
import pandas as pd
import pdb

class CSVParser(data.Dataset):
    def __init__(self, X_file, params):
        super(CSVParser, self).__init__()
        self.sep = params["sep"]
        self.header = params["header"]
        self.X = pd.read_csv(X_file, sep=self.sep, header=self.header)
        self.label_header = params["label_header"]
        self.labels =  self.X[self.label_header]
        if "normalizer_const" in params:
            self.X = self.X / params["normalizer_const"]
        del self.X[self.label_header]

        self.length = self.X.shape[0]
        self.dimension = self.X.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = self.labels[index]
        data_point = np.array(self.X.loc[index].values)
        return data_point, label

