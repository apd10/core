import torch
torch.manual_seed(0)
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from os import path
import os

import pdb
import argparse
from os.path import dirname, abspath, join
import glob
cur_dir = dirname(abspath(__file__))
import yaml
from Loop import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', action="store", dest="config", type=str, default=None, required=True,
                    help="config to setup the training")

results = parser.parse_args()
config_file = results.config
with open(config_file, "r") as f:
  config = yaml.load(f)
run = Loop(config)
run.loop()
