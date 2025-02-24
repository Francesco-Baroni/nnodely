import torch
import torch.nn as nn

from nnodely import Parameter, Constant
from nnodely.relation import Stream, ToStream, toStream
from nnodely.model import Model
from nnodely.utils import check, enforce_types


relu_relation_name = 'ReLU'
tanh_relation_name = 'Tanh'
elu_relation_name = 'ELU'

class Relu(Stream, ToStream):
    """
        Implement the Rectified-Linear Unit (ReLU) relation function.

        See also:
            Official PyTorch ReLU documentation: 
            `torch.nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_

        :param obj: The relation stream.
        :type obj: Stream 

        Example:
            >>> x = Relu(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|float|int) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Relu operation.")
        super().__init__(relu_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [relu_relation_name,[obj.name]]

class Tanh(Stream, ToStream):
    """
        Implement the Hyperbolic Tangent (Tanh) relation function.

        See also:
            Official PyTorch tanh documentation: 
            `torch.nn.Tanh <https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html>`_

        :param obj: The relation stream.
        :type obj: Stream 

        Example:
            >>> x = Tanh(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|float|int) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream,TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Tanh operation.")
        super().__init__(tanh_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [tanh_relation_name,[obj.name]]

class ELU(Stream, ToStream):
    """
        Implement the Exponential-Linear Unit (ELU) relation function.

        See also:
            Official PyTorch ReLU documentation: 
            `torch.nn.ELU <https://pytorch.org/docs/stable/generated/torch.nn.ELU.html>`_

        :param obj: The relation stream.
        :type obj: Stream 

        Example:
            >>> x = ELU(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|float|int) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream,TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Tanh operation.")
        super().__init__(elu_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [elu_relation_name,[obj.name]]

class Tanh_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self,):
        super(Tanh_Layer, self).__init__()
    def forward(self, x):
        return torch.tanh(x)

def createTanh(self, *input):
    """
     :noindex:
    """
    return Tanh_Layer()

class ReLU_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self,):
        super(ReLU_Layer, self).__init__()
    def forward(self, x):
        return torch.relu(x)
    
def createRelu(self, *input):
    """
     :noindex:
    """
    return ReLU_Layer()
    

def createELU(self, *input):
    """
     :noindex:
    """
    return nn.ELU()

setattr(Model, relu_relation_name, createRelu)
setattr(Model, tanh_relation_name, createTanh)
setattr(Model, elu_relation_name, createELU)