import numpy as np
import torch
from torch.utils import data
class GenSVMFormatParser(data.Dataset):
    def __init__(self, X_file, params):
        super(GenSVMFormatParser, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        self.length = len(self.X)
        self.dim = params["dimension"]
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        data = self.X[index].strip().split(" ")

        label = int(data[0])
        xdata = data[1:]

        for xd in xdata:
            temp = xd.split(":")
            data_point[int(temp[0])] = float(temp[1])
        return data_point, label

