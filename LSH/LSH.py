import numpy as np


# TODO: GET X ID
def get_x_id(x):
    return x


class LSH:
    def __init__(self, r, l, k, n):
        self.R = r
        self.L = l
        self.K = k
        self.data = np.zeros(n)
        self.sketch = np.zeros((r, l))
        self.base_idx = np.zeros(r * l + 1)
        self.lsh_fn = []
        self.query_results = []

    # TODO: GENERATE LSH FN
    def generate_lsh_fn(self):
        self.lsh_fn = []

    def get_lsh_fn(self):
        return self.lsh_fn

    def import_lsh_fn(self, new_lsh_fn):
        self.lsh_fn = new_lsh_fn

    def __hash(self, x, fn):
        for i in range(self.L):
            h = self.lsh_fn[i](x)
            fn(i, h, x)

    def __insert_to_sketch(self, i, h):
        self.sketch[i, h] += 1

    def __make_sketch(self, dataset):
        for x in dataset:
            self.__hash(x, self.__insert_to_sketch)

    def __make_base_idx(self):
        for i in range(self.L):
            for j in range(self.R):
                idx = i * self.R + j
                self.base_idx[idx + 1] = self.base_idx[idx] + self.sketch[i, j]

    def __zero_sketch(self):
        self.sketch = np.zeros((self.R, self.L))

    def __insert_to_data(self, i, h, x):
        idx = self.base_idx[i * self.R + h] + self.sketch[i, h]
        self.data[idx] = get_x_id(x)
        self.__insert_to_sketch(i, h)

    def __make_data(self, dataset):
        for x in dataset:
            self.__hash(x, self.__insert_to_data)

    def __get_bucket_contents(self, i, h):
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

