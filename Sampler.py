from samplers.list_samplers import *

class Sampler:
    def get(dataset, name, params):
        if name == "simple":
            sampler = SimpleSampler(dataset, params)
        elif name == "subsimple":
            sampler = SubSimpleSampler(dataset, params)
        else:
            raise NotImplementedError
        return sampler

        
      
