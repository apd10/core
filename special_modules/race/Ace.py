import pdb
from special_modules.cms.CS import *

class Ace():
    def get(ds_type, seed, params):
        if ds_type == "array":
            ace = AceArray(seed, params)
        elif ds_type == "cs":
            ace = AceCS(seed, params)
        else:
            raise NotImplementedError
        return ace


class AceArray():
    def __init__(self, seed, params):
        raise NotImplementedError

    def insert(self, keys, values):
        raise NotImplementedError

    def get_top_buckets(self):
        raise NotImplementedError

class AceCS():
    def __init__(self, seed, params):
        self.countsketch = CountSketch(seed, params)

    def insert(self, keys, values): # b x num_keys
        self.countsketch.insert(keys, values)

    def get_top_buckets(self):
        return self.countsketch.get_top()
    
    def query(self, keys): # b x num_keys
        return self.countsketch.query(keys)

    def get_dictionary(self):
        return self.countsketch.get_dictionary()

    def set_dictionary(self, dic):
        return self.countsketch.set_dictionary(dic)
