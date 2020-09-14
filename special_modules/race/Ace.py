import pdb
from special_modules.cms.CS import *

class Ace():
    def get(ds_type, seed, params, norm_info):
        if ds_type == "array":
            ace = AceArray(seed, params, norm_info)
        elif ds_type == "cs":
            ace = AceCS(seed, params, norm_info)
        else:
            raise NotImplementedError
        return ace


class AceArray():
    def __init__(self, seed, params, norm_info):
        raise NotImplementedError

    def insert(self, keys, values):
        raise NotImplementedError

    def get_top_buckets(self):
        raise NotImplementedError


class AceCS():
    def __init__(self, seed, params, norm_info):
        self.countsketch = CountSketch(seed, params)
        self.store_norms = False
        if norm_info :
            self.store_norms = True
            self.normsketch = CountSketch(seed, params) # same count sketch . However it will store the sum of norms in that bucket
            

    def insert(self, keys, values, norm_values=None): # b x num_keys
        self.countsketch.insert(keys, values)
        if self.store_norms:
            assert(norm_values is not None)
            self.normsketch.insert(keys, norm_values)


    def get_top_buckets(self):
        top_df =  self.countsketch.get_top()
        if self.store_norms:
            assert("norm_sum" in top_df.columns)
        return top_df
    
    def query(self, keys): # b x num_keys
        counts = self.countsketch.query(keys)
        if self.store_norms:
            norms_sums = self.normsketch.query(keys)
            return counts, norms_sums
        else:
            return counts

    def get_dictionary(self):
        dic =  self.countsketch.get_dictionary()
        if "topK" in dic and self.store_norms:
            # append the norm information in the 
            cols = [a for a in dic["topK"].columns if a != "value"]
            hashes = torch.LongTensor(dic["topK"][cols].values)
            norms = self.normsketch.query(hashes)
            dic["topK"]["norm_sum"] = np.array(norms.cpu())
        return dic

    def set_dictionary(self, dic):
        return self.countsketch.set_dictionary(dic)

    def onDecay(self, samples):
        self.countsketch.onDecay(samples)
