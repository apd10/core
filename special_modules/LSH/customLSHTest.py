import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from LSH.RushLSH import RushLSH
import flamegraph

class GetXIdFactory:
    def __init__(self):
        self.curr_id = 0

    def get_x_id(self, x):
        _id = self.curr_id
        self.curr_id += 1
        return _id


factory = GetXIdFactory()


def data_to_vector(data):
    images, _ = data
    return images.view(-1, 28 * 28)[0].numpy()


LSH = RushLSH(28*28, 100, 10, 10, to_vector=lambda v: v, get_x_id=factory.get_x_id)

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
train_dataset = [data_to_vector(data) for data in train_loader]
LSH.build(train_dataset)

train_end = time.time()
train_time = train_end - train_start
print(f'Trained with {len(train_loader)} images in {train_time} seconds')
print(f'Throughput: {len(train_loader) / train_time} images per second')
print(f'{train_time / len(train_loader)} seconds per image')


# test_start = time.time()
#
# n_test = 0
# for q in test_loader:
#     n_test += 1
#     result = LSH.query(q)
#     # print(result)
#
# test_end = time.time()
# test_time = test_end - test_start
# print(f'Queried {n_test} images in {test_time} seconds')
# print(f'Throughput: {n_test / test_time} queries per second')
# print(f'{test_time / n_test} seconds per query')




