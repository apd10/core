import time
import numpy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          shuffle=False)

train_start = time.time()

# Dimension of our vector space
dimension = 784

# Create a random binary hash with 10 bits
rbp = RandomBinaryProjections('rbp', 10)

# Create engine with pipeline configuration
engine = Engine(dimension, lshashes=[rbp])

# Index training mnist vectors
index = 0
for images, labels in train_loader:
    v = images.view(-1, 28 * 28)[0].numpy()
    engine.store_vector(v, 'data_%d' % index)
    index += 1

train_end = time.time()
train_time = train_end - train_start
print(f'Trained with {index} images in {train_time} seconds')
print(f'Throughput: {index / train_time} images per second')
print(f'{train_time / index} seconds per image')

test_start = time.time()

n_test = 0
for images, labels in test_loader:
    q = images.view(-1, 28 * 28)[0].numpy()
    N = engine.neighbours(q)
    n_test += 1

test_end = time.time()
test_time = test_end - test_start
print(f'Queried {n_test} images in {test_time} seconds')
print(f'Throughput: {n_test / test_time} queries per second')
print(f'{test_time / n_test} seconds per query')


