import torch
import numpy as np
from numpy.linalg import norm
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from nearpy import Engine
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjections
from nearpy.storage import MemoryStorage
import time
from skopt import gp_minimize


# LOAD DATA
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers)
# data loader for hashing the training data; load single data points instead of in batches
train_loader_lsh = torch.utils.data.DataLoader(test_data)

i = 0


def loss_grad_inv(w, x, n_classes, c):
    classes = np.arange(n_classes)
    classes_except_c = np.array([i for i in range(n_classes) if i != c])

    def exw(i):
        return np.exp([np.matmul(w[i].detach().numpy(), x)])
    exw_j = np.sum(exw(classes))

    def i_neq_c(i):
        return (exw(i) / exw_j) ** 2
    first_inner_term = np.sum(i_neq_c(classes_except_c))
    second_inner_term = ((exw(c) / exw_j - 1) ** 2)[0]
    loss = (norm(x) ** 2) * (first_inner_term + second_inner_term)
    # print(loss, flush=True)
    return 1 / loss


def optimization_domain(dim, domain):
    # print("and another c", flush=True)
    return np.array([domain for _ in range(dim)])


# DEFINE NETWORK ARCHITECTURE
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        return x


# initialize the NN
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.train()

# PREPARE LSH
dimension = 784
# Prepare LSH functions
K = 10  # LSH concatenations
L = 100  # LSH arrays
rbp = [RandomBinaryProjections('rbp', K) for i in range(L)]
# Prepare LSH table
storage = MemoryStorage()
engine = Engine(dimension, lshashes=rbp, storage=storage, vector_filters=[NearestFilter(1)])

i = 0
# Hash data
for data, target in train_loader_lsh:
    v = data.view(-1, 28 * 28)[0].numpy()
    # if i == 0:
        # print(v)
    engine.store_vector(v, target)
    i += 1

############
# TRAINING #
############
n_epochs = 1
train_start = time.time()
for epoch in range(n_epochs):
    epoch_start = time.time()
    # monitor training loss
    train_loss = 0.0
    # track batch with maximum gradient
    max_grad = 00
    # max_grad_batch = None

    # # Find batch with highest gradient
    # for data, target in train_loader:
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     # calculate gradient and adjust max grad
    #     grad = np.linalg.norm(model.fc1.weight.grad.numpy())
    #     if grad > max_grad:
    #         max_grad = grad
    #         max_grad_batch = data
    def loss_grad_w(c):
        return lambda x: loss_grad_inv(model.fc1.weight, x, 10, c)


    # print("getting max_grad_batch", flush=True)
    max_grad_batch = [gp_minimize(loss_grad_w(c), optimization_domain(28 * 28, (0, 1)), n_calls=10) for c in range(10)]
    # print(max_grad_batch, flush=True)
    # Collect neighbors of data in the batch with maximum gradient
    neighbors = []
    i = 0
    for m in max_grad_batch:
        # if i == 0:
        #     print(np.array(m["x"]))
        # v = d.view(-1, 28 * 28)[0].numpy()
        neighbors.extend(engine.neighbours(np.array(m["x"])))
        i += 1
    print(len(neighbors))

    # Reformat to be compatible with data loader
    neighbors = [(torch.from_numpy(v).view(1, 28, 28).float(), data.numpy()[0]) for (v, data, distance) in neighbors]
    train_loader_2 = torch.utils.data.DataLoader(neighbors, batch_size=batch_size,
                                                 num_workers=num_workers)

    # Optimize weights only with neighbors of batch with highest gradient
    for data, target in train_loader_2:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    epoch_end = time.time()
    print(epoch_end - epoch_start, "seconds for this epoch")

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader_2.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch + 1,
        train_loss
    ))

train_end = time.time()
train_time = train_end - train_start
print(f'Trained in {train_time} seconds')


# TEST NETWORK
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


