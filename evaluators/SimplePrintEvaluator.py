import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class SimplePrintEvaluator:
    def __init__(self, params, train_data, test_data, device_id):
      self.train_data = train_data
      self.test_data = test_data
      self.device_id = device_id
      self.eval_itr = params["eval_itr"]
      self.eval_epoch = params["eval_epoch"]
      self.skip_0 = False
      if "skip_0" in params:
          self.skip_0  = params["skip_0"]
      self.csv_dump = None
      if "csv_dump" in params:
          self.csv_dump = params["csv_dump"]

    def evaluate(self, epoch, loc_itr, iteration, model, loss_func): # also logs
        if self.skip_0 :
            if epoch == 0 and loc_itr == 0:
                return
        if iteration % self.eval_itr == 0 or (epoch % self.eval_epoch == 0 and loc_itr == 0):
            #self._eval(epoch, loc_itr, iteration, model, loss_func, self.train_data, "TRAIN")
            self._eval(epoch, loc_itr, iteration, model, loss_func, self.test_data, "TEST")

    def _eval(self, epoch, loc_itr, iteration, model, loss_func, data, key):
        count = 0
        total_loss = 0.
        model.eval()
        correct = 0.0
        with torch.no_grad():
            data.reset()
            while not data.end():
                x, y = data.next()
                x = Variable(x).cuda(self.device_id) if self.device_id!=-1 else Variable(x)
                y = Variable(y).cuda(self.device_id) if self.device_id!=-1 else Variable(y)
                output = model(x)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()
                loss = loss_func(output, y, size_average=False).item()
                total_loss += loss
                count += 1
        valid_loss = total_loss / data.len()
        acc = correct / data.len()
        print('{} : Epoch : {} Loc_itr: {} Iteration: {} Loss: {} Accuracy: {}'.format(key, epoch, loc_itr, iteration, valid_loss, acc))
        if self.csv_dump is not None:
            f = open(self.csv_dump, "a")
            if iteration == 0:
                f.write('{},{},{},{},{},{}\n'.format("key", "epoch", "loc_itr", "iteration", "loss", "acc"))
            f.write('{},{},{},{},{},{}\n'.format(key, epoch, loc_itr, iteration, valid_loss, acc))
            f.close()
            
