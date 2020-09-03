import numpy as np
import torch
from torch.utils import data
import pdb

class GenSVMFormatParser(data.Dataset):
    def __init__(self, X_file, params):
        super(GenSVMFormatParser, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        self.length = len(self.X)
        self.dim = params["dimension"]
        self.base_idx = 0
        if "base_idx" in params:
            self.base_idx = params["base_idx"]

        self.class_base_idx = 0
        if "class_base_idx" in params:
            self.class_base_idx = params["class_base_idx"]
        self.normalizer_const = 1
        if "normalizer_const" in params:
            self.normalizer_const = params["normalizer_const"]

        self.neg_class = False
        if "neg_class" in params:
            self.neg_class = True


    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        data = self.X[index].strip().split(" ")

        label = int(data[0]) - self.class_base_idx
        if self.neg_class and label == -1:
            label = 0
            
        xdata = data[1:]

        for xd in xdata:
            temp = xd.split(":")
            data_point[int(temp[0]) - self.base_idx] = float(temp[1]) / self.normalizer_const
        return data_point, label

