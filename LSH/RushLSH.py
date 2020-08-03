import numpy as np
from nearpy.hashes import RandomBinaryProjections
import mmh3


class RushLSH:
    def __init__(self, dimension, r, l, k, to_vector=None, get_x_id=None):
        self.dim = dimension
        self.R = r
        self.L = l
        self.K = k
        self.N = 0
        self.sketch = np.zeros((l, r), dtype=int)
        self.base_idx = np.zeros(l * r + 1, dtype=int)
        self.data = np.array([], dtype=int)
        self.lsh_fn = []
        self.query_results = []
        self.to_vector = to_vector
        if self.to_vector is None:
            self.to_vector = lambda v: v
        self.get_x_id = get_x_id
        if self.get_x_id is None:
            self.get_x_id = lambda x: x

    def generate_lsh_fn(self):
        self.lsh_fn = []
        for i in range(self.L):
            rbp = RandomBinaryProjections('rbp', self.K)
            rbp.reset(self.dim)

            # def fn(x):
            #     mm = mmh3.hash(rbp.hash_vector(x)[0])
            #     return mm % self.R
            def fn(x):
                return 1
            self.lsh_fn.append(fn)

    def get_lsh_fn(self):
        return self.lsh_fn

    def import_lsh_fn(self, new_lsh_fn):
        self.lsh_fn = new_lsh_fn

    def __hash(self, x, fn):
        x_vector = self.to_vector(x)
        x_id = self.get_x_id(x_vector)
        for i in range(self.L):
            h = self.lsh_fn[i](x_vector)
            fn(i, h, x_id)

    def __insert_to_sketch(self, i, h, x_id):
        self.sketch[i, h] += 1

    def __make_sketch(self, dataset):
        for x in dataset:
            self.N += 1
            self.__hash(x, self.__insert_to_sketch)

    def __make_base_idx(self):
        for i in range(self.L):
            for j in range(self.R):
                idx = i * self.R + j
                self.base_idx[idx + 1] = self.base_idx[idx] + self.sketch[i, j]

    def __zero_sketch(self):
        self.sketch = np.zeros((self.L, self.R), dtype=int)

    def __insert_to_data(self, i, h, x_id):
        idx = self.base_idx[i * self.R + h] + self.sketch[i, h]
        self.data[idx] = self.get_x_id(x_id)
        self.__insert_to_sketch(i, h, x_id)

    def __make_data(self, dataset):
        self.data = np.zeros(self.N * self.L, dtype=int)
        for x in dataset:
            self.__hash(x, self.__insert_to_data)

    def __get_bucket_contents(self, i, h, x_id):
        idx = i * self.R + h
        self.query_results.extend(self.data[self.base_idx[idx]:self.base_idx[idx + 1]])

    def build(self, dataset):
        self.generate_lsh_fn()
        self.__make_sketch(dataset)
        self.__make_base_idx()
        self.__zero_sketch()
        self.__make_data(dataset)

    def query(self, q):
        self.query_results = []
        self.__hash(q, self.__get_bucket_contents)
        return self.query_results


