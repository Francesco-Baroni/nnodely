import torch.nn as nn
import torch

from nnodely.relation import ToStream, Stream, toStream
from nnodely.model import Model
from nnodely.utils import check, merge, enforce_types
from nnodely.parameter import Parameter, Constant

# Binary operators
add_relation_name = 'Add'
sub_relation_name = 'Sub'
mul_relation_name = 'Mul'
div_relation_name = 'Div'
pow_relation_name = 'Pow'

# Unary operators
neg_relation_name = 'Neg'

# Merge operator
sum_relation_name = 'Sum'
class Add(Stream, ToStream):
    """
        Implement the addition function between two tensors. 
        (it is also possible to use the classical math operator '+')

        See also:
            Official PyTorch Add documentation: 
            `torch.add <https://pytorch.org/docs/stable/generated/torch.add.html>`_

        :param input1: the first element of the addition
        :type obj: Tensor
        :param input2: the second element of the addition
        :type obj: Tensor

        Example:
            >>> add = Add(relation1, relation2)
            or
            >>> add = relation1 + relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1,obj2 = toStream(obj1),toStream(obj2)
        check(type(obj1) is Stream,TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for add operation.")
        check(type(obj2) is Stream,TypeError,
              f"The type of {obj2} is {type(obj2)} and is not supported for add operation.")
        check(obj1.dim == obj2.dim or obj1.dim == {'dim':1} or obj2.dim == {'dim':1}, ValueError,
              f"For addition operators (+) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(add_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [add_relation_name,[obj1.name,obj2.name]]

## TODO: check the scalar dimension, helpful for the offset
class Sub(Stream, ToStream):
    """
        Implement the subtraction function between two tensors. 
        (it is also possible to use the classical math operator '-')

        :param input1: the first element of the subtraction
        :type obj: Tensor
        :param input2: the second element of the subtraction
        :type obj: Tensor

        Example:
            >>> sub = Sub(relation1, relation2)
            or
            >>> sub = relation1 - relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2 = toStream(obj1), toStream(obj2)
        check(type(obj1) is Stream,TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for sub operation.")
        check(type(obj2) is Stream,TypeError,
              f"The type of {obj2} is {type(obj2)} and is not supported for sub operation.")
        check(obj1.dim == obj2.dim or obj1.dim == {'dim':1} or obj2.dim == {'dim':1}, ValueError,
              f"For subtraction operators (-) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(sub_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [sub_relation_name,[obj1.name,obj2.name]]

class Mul(Stream, ToStream):
    """
        Implement the multiplication function between two tensors. 
        (it is also possible to use the classical math operator '*')

        :param input1: the first element of the multiplication
        :type obj: Tensor
        :param input2: the second element of the multiplication
        :type obj: Tensor

        Example:
            >>> mul = Mul(relation1, relation2)
            or
            >>> mul = relation1 * relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2 = toStream(obj1), toStream(obj2)
        check(type(obj1) is Stream, TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for mul operation.")
        check(type(obj2) is Stream, TypeError,
              f"The type of {obj2} is {type(obj2)} and is not supported for mul operation.")
        check(obj1.dim == obj2.dim or obj1.dim == {'dim':1} or obj2.dim == {'dim':1}, ValueError,
              f"For multiplication operators (*) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(mul_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [mul_relation_name,[obj1.name,obj2.name]]

class Div(Stream, ToStream):
    """
        Implement the division function between two tensors. 
        (it is also possible to use the classical math operator '/')

        :param input1: the numerator of the division
        :type obj: Tensor
        :param input2: the denominator of the division
        :type obj: Tensor

        Example:
            >>> div = Div(relation1, relation2)
            or
            >>> div = relation1 / relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2 = toStream(obj1), toStream(obj2)
        check(type(obj1) is Stream, TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for div operation.")
        check(type(obj2) is Stream, TypeError,
              f"The type of {obj2} is {type(obj2)} and is not supported for div operation.")
        check(obj1.dim == obj2.dim or obj1.dim == {'dim':1} or obj2.dim == {'dim':1}, ValueError,
              f"For division operators (*) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(div_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [div_relation_name,[obj1.name,obj2.name]]

class Pow(Stream, ToStream):
    """
        Implement the power function given an input and an exponent. 
        (it is also possible to use the classical math operator '**')

        See also:
            Official PyTorch Add documentation: 
            `torch.add <https://pytorch.org/docs/stable/generated/torch.pow.html>`_

        :param input: the base of the power function
        :type obj: Tensor
        :param exp: the exponent of the power function
        :type obj: float or Tensor

        Example:
            >>> pow = Pow(relation, exp)
            or
            >>> pow = relation1 ** relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2 = toStream(obj1), toStream(obj2)
        check(type(obj1) is Stream, TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for exp operation.")
        check(type(obj2) is Stream, TypeError,
              f"The type of {obj2} is {type(obj2)} but must be int or float and is not supported for exp operation.")
        check(obj1.dim == obj2.dim or obj1.dim == {'dim':1} or obj2.dim == {'dim':1}, ValueError,
              f"For division operators (*) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(pow_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [pow_relation_name,[obj1.name,obj2.name]]

class Neg(Stream, ToStream):
    """
        Implement the negate function given an input. 

        :param input: the input to negate
        :type obj: Tensor

        Example:
            >>> x = Neg(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for neg operation.")
        super().__init__(neg_relation_name+str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [neg_relation_name,[obj.name]]

class Sum(Stream, ToStream):
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for sum operation.")
        super().__init__(sum_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [sum_relation_name,[obj.name]]

class Add_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Add_Layer, self).__init__()

    def forward(self, *inputs):
        results = inputs[0]
        for input in inputs[1:]:
            results = results + input
        return results
        #return torch.add(inputs[0],inputs[1]))
        #return torch.sum(torch.stack(list(inputs)),dim=0)

def createAdd(name, *inputs):
    #: :noindex:
    return Add_Layer()

class Sub_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Sub_Layer, self).__init__()

    def forward(self, *inputs):
        # Perform element-wise subtraction
        results = inputs[0]
        for input in inputs[1:]:
            results = results - input
        return results
        #return torch.add(inputs[0], -inputs[1])
        #return torch.add(inputs[0],-torch.sum(torch.stack(list(inputs[1:])),dim=0))

def createSub(self, *inputs):
    #: :noindex:
    return Sub_Layer()


class Mul_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Mul_Layer, self).__init__()

    def forward(self, *inputs):
        results = inputs[0]
        for input in inputs[1:]:
            results = results * input
        return results
        #return inputs[0] * inputs[1]
        #return torch.prod(torch.stack(list(inputs)),dim=0)


def createMul(name, *inputs):
    #: :noindex:
    return Mul_Layer()

class Div_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Div_Layer, self).__init__()

    def forward(self, *inputs):
        results = inputs[0]
        for input in inputs[1:]:
            results = results / input
        return results
        #return inputs[0] / inputs[1]
        #return inputs[0] / torch.prod(torch.stack(list(inputs[1:])),dim=0)

def createDiv(name, *inputs):
    #: :noindex:
    return Div_Layer()

class Pow_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Pow_Layer, self).__init__()

    def forward(self, *inputs):
        return torch.pow(inputs[0], inputs[1])

def createPow(name, *inputs):
    #: :noindex:
    return Pow_Layer()

class Neg_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Neg_Layer, self).__init__()

    def forward(self, x):
        return -x

def createNeg(self, *inputs):
    #: :noindex:
    return Neg_Layer()

class Sum_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Sum_Layer, self).__init__()

    def forward(self, inputs):
        return torch.sum(inputs, dim = 2)

def createSum(name, *inputs):
    #: :noindex:
    return Sum_Layer()

setattr(Model, add_relation_name, createAdd)
setattr(Model, sub_relation_name, createSub)
setattr(Model, mul_relation_name, createMul)
setattr(Model, div_relation_name, createDiv)
setattr(Model, pow_relation_name, createPow)

setattr(Model, neg_relation_name, createNeg)

setattr(Model, sum_relation_name, createSum)


