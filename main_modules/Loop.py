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

class Loop:
    def __init__(self, params):
        self.device_id = params["device_id"]
        self.epochs = params["epochs"]
        # data
        self.train_data = Data(params["train_data"])
        #self.test_data = Data(params["test_data"])
        #self.validation_data = Data(params["validation_data"])
        #self.progress_train_data = Data(params["progress_train_data"])
        self.progress_train_data = None
        self.progress_test_data = Data(params["progress_test_data"])
        # model
        self.model = Model.get(params["model"])
        print(self.model)
        if self.device_id != -1:
          self.model = self.model.cuda(self.device_id)
        # optimizer
        self.optimizer = Optimizer.get(self.model, params["optimizer"])
        # loss
        self.loss_func = Loss.get(params["loss"])
        #if self.device_id != -1:
        #  self.loss_func = self.loss_func.cuda(self.device_id)
        # progress evaluator
        self.progress_evaluator = ProgressEvaluator.get(params["progress_evaluator"], self.progress_train_data, self.progress_test_data, self.device_id)

    def loop(self):
        epoch = 0
        iteration = 0

        while epoch < self.epochs :
            self.train_data.reset()
            num_samples = self.train_data.len()
            batch_size = self.train_data.batch_size()
            num_batches = int(np.ceil(num_samples/batch_size))
            loc_itr = 0
            for i in tqdm(range(num_batches)):
                if self.train_data.end():
                  break
                self.model.train()
                self.optimizer.zero_grad()
                x, y = self.train_data.next()
                x = Variable(x).cuda(self.device_id) if self.device_id!=-1 else Variable(x)
                y = Variable(y).cuda(self.device_id) if self.device_id!=-1 else Variable(y)
                output = self.model(x)
                loss = self.loss_func(output, y)
                loss.backward()
                self.optimizer.step()
                self.progress_evaluator.evaluate(epoch, loc_itr, iteration, self.model, self.loss_func)
                iteration = iteration + 1
                loc_itr = loc_itr + 1
                #print("Loss", loss)
            epoch = epoch + 1
