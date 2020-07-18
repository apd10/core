from models.list_models import *

class Model:
  def get(params):
    model = None
    if params["name"] == "MLP":
      model = FCN(params["MLP"]["input_dim"], params["MLP"]["num_layers"], params["MLP"]["hidden_size"], params["MLP"]["num_class"])
    else:
      raise NotImplementedError
    return model


