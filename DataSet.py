from dataparsers.list_dataparsers import *


class DataSet:
    ''' specifies the parsing for the data
    '''
    def get(data_file, name, params):
        if name == "gensvm":
          dataset = GenSVMFormatParser(data_file, params)
        elif name == "csv":
          dataset = CSVParser(data_file, params)
        else:
          raise NotImplementedError
        return dataset


