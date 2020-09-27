import torch
from torch import nn
from torch.autograd import Variable
from Data import Data
from Model import Model
from Optimizer import Optimizer
from Loss import Loss
from ProgressEvaluator import ProgressEvaluator
import pdb
import numpy as np
from tqdm import tqdm

class Stats:
    def __init__(self, params):
        self.device_id = params["device_id"]
        self.data = Data(params["data"])
        self.stat_file = params["stat_file"]

    def run(self):
        device_id = self.device_id
        num_samples = self.data.len()
        batch_size = self.data.batch_size()
        num_batches = int(np.ceil(num_samples/batch_size))
        loc_itr = 0
        mean = None
        mean_sum = None
        var = None
        var_sum = None
        
        self.data.reset()
        for i in tqdm(range(num_batches)):
            if self.data.end():
                break
            x, y = self.data.next()
            x = x.cuda(device_id)
            xsum = torch.sum(x, dim=0)
            if mean_sum is None:
                mean_sum = xsum
            else:
                mean_sum += xsum

        mean = mean_sum / num_samples

        self.data.reset()
        for i in tqdm(range(num_batches)):
            if self.data.end():
                break
            x, y = self.data.next()
            x = x.cuda(device_id)
            xcent = x - mean
    
            xsum = torch.sum(torch.mul(xcent, xcent), dim=0)
            if var_sum is None:
                var_sum = xsum
            else:
                var_sum += xsum

        var = var_sum / num_samples
        std = torch.sqrt(var)
        

        mean = np.array(mean.cpu())
        std = np.array(std.cpu())
        np.savez_compressed(self.stat_file, mu=mean, std=std)
