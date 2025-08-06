import torch
import torch.nn as nn

from nnodely.basic.relation import ToStream, Stream, toStream
from nnodely.basic.model import Model
from nnodely.support.utils import check, enforce_types
from nnodely.layers.parameter import Parameter, Constant
from nnodely.support.jsonutils import merge, binary_cheks

sin_relation_name = 'Sin'
cos_relation_name = 'Cos'
tan_relation_name = 'Tan'
tanh_relation_name = 'Tanh'
cosh_relation_name = 'Cosh'
sech_relation_name = 'Sech'
atan2_relation_name = 'Atan2'

class Sin(Stream, ToStream):
    """
    Implement the sine function given an input relation.

    See also:
        Official PyTorch Sin documentation: 
        `torch.sin <https://pytorch.org/docs/stable/generated/torch.sin.html>`_

    :param obj: the input relation stream
    :type obj: Stream

    Example:
        >>> sin = Sin(relation)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|int|float) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Sin operation.")
        super().__init__(sin_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [sin_relation_name, [obj.name]]

class Cos(Stream, ToStream):
    """
    Implement the cosine function given an input relation.

    See also:
        Official PyTorch Cos documentation: 
        `torch.cos <https://pytorch.org/docs/stable/generated/torch.cos.html>`_

    :param obj: the input relation stream
    :type obj: Stream

    Example:
        >>> cos = Cos(relation)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|int|float) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Cos operation.")
        super().__init__(cos_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [cos_relation_name, [obj.name]]

class Tan(Stream, ToStream):
    """
    Implement the tangent function given an input relation.

    See also:
        Official PyTorch Tan documentation: 
        `torch.tan <https://pytorch.org/docs/stable/generated/torch.tan.html>`_

    :param obj: the input relation stream
    :type obj: Stream

    Example:
        >>> tan = Tan(relation)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|int|float) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Tan operation.")
        super().__init__(tan_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [tan_relation_name, [obj.name]]

class Cosh(Stream, ToStream):
    """
    Returns a new tensor with the hyperbolic cosine of the elements of input.

    See also:
        Official PyTorch Cosh documentation: 
        `torch.cosh <https://pytorch.org/docs/stable/generated/torch.cosh.html>`_

    :param obj: the input relation stream
    :type obj: Stream

    Example:
        >>> cosh = Cosh(relation)
    """
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Cosh operation.")
        super().__init__(cosh_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [cosh_relation_name, [obj.name]]

class Sech(Stream, ToStream):
    """
    Returns a new tensor with the hyperbolic secant of the elements of input.

    :param obj: the input relation stream
    :type obj: Stream

    Example:
        >>> sech = Sech(relation)
    """
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Sech operation.")
        super().__init__(sech_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [sech_relation_name, [obj.name]]

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

class Atan2(Stream, ToStream):
    """
        Implement the arctangent of y/x. 

        See also:
            Official PyTorch pow documentation:
            `torch.atan2 <https://docs.pytorch.org/docs/stable/generated/torch.atan2.html>`_

        :param y: the base of the power function
        :type obj: Tensor
        :param x: the exponent of the power function
        :type obj: float or Tensor

        Example:
            >>> atan2 = Atan2(y, x)
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = binary_cheks(self, obj1, obj2, 'atan2 operators')
        super().__init__(atan2_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [atan2_relation_name,[obj1.name,obj2.name]]

class Sin_Layer(nn.Module):
    def __init__(self,):
        super(Sin_Layer, self).__init__()
    def forward(self, x):
        return torch.sin(x)

def createSin(self, *inputs):
    return Sin_Layer()

class Cos_Layer(nn.Module):
    def __init__(self,):
        super(Cos_Layer, self).__init__()
    def forward(self, x):
        return torch.cos(x)

def createCos(self, *inputs):
    return Cos_Layer()

class Tan_Layer(nn.Module):
    def __init__(self,):
        super(Tan_Layer, self).__init__()
    def forward(self, x):
        return torch.tan(x)

def createTan(self, *inputs):
    return Tan_Layer()

class Cosh_Layer(nn.Module):
    def __init__(self,):
        super(Cosh_Layer, self).__init__()
    def forward(self, x):
        return torch.cosh(x)

def createCosh(self, *inputs):
    return Cosh_Layer()

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

class Sech_Layer(nn.Module):
    def __init__(self,):
        super(Sech_Layer, self).__init__()
    def forward(self, x):
        return 1/torch.cosh(x)

def createSech(self, *inputs):
    return Sech_Layer()

class Atan2_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Atan2_Layer, self).__init__()

    def forward(self, *inputs):
        return torch.atan2(inputs[0], inputs[1])

def createAtan2(name, *inputs):
    #: :noindex:
    return Atan2_Layer()

setattr(Model, sin_relation_name, createSin)
setattr(Model, cos_relation_name, createCos)
setattr(Model, tan_relation_name, createTan)
setattr(Model, cosh_relation_name, createCosh)
setattr(Model, tanh_relation_name, createTanh)
setattr(Model, sech_relation_name, createSech)
setattr(Model, atan2_relation_name, createAtan2)