import numpy as np
import torch
from torch.utils import data
import pandas as pd
import pdb

class CSVParser(data.Dataset):
    def __init__(self, X_file, params):
        super(CSVParser, self).__init__()
        self.sep = params["sep"]
        self.header = None
        if "header" in params:
            self.header = params["header"]
        self.skiprows = None
        if "skiprows" in params:
            self.skiprows = params["skiprows"]
        self.X = pd.read_csv(X_file, sep=self.sep, header=self.header, skiprows=self.skiprows)

        self.label_header = None
        if "label_header" in params:
            self.label_header = params["label_header"]
            self.labels =  self.X[self.label_header]
            del self.X[self.label_header]
        else:
            self.labels = np.zeros(self.X.shape[0])

        if "ignore_label" in params:
            self.labels = np.zeros(self.X.shape[0])

        if "normalizer_const" in params:
            self.X = self.X / params["normalizer_const"]

        self.length = self.X.shape[0]
        self.dimension = self.X.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = self.labels[index]
        data_point = np.array(self.X.loc[index].values)
        return data_point, label

