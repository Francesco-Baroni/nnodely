import torch.nn as nn
import torch

from nnodely.basic.relation import ToStream, Stream, toStream
from nnodely.basic.model import Model
from nnodely.support.utils import check, merge, enforce_types, get_window
from nnodely.layers.parameter import Parameter, Constant

# Binary operators
add_relation_name = 'Add'
sub_relation_name = 'Sub'
mul_relation_name = 'Mul'
div_relation_name = 'Div'
pow_relation_name = 'Pow'
atan2_relation_name = 'Atan2'

# Unary operators
neg_relation_name = 'Neg'
sign_relation_name = 'Sign'

# Merge operator
sum_relation_name = 'Sum'

def aritmetic_cheks(self, obj1, obj2, name):
    obj1,obj2 = toStream(obj1),toStream(obj2)
    check(type(obj1) is Stream,TypeError,
          f"The type of {obj1} is {type(obj1)} and is not supported for add operation.")
    check(type(obj2) is Stream,TypeError,
          f"The type of {obj2} is {type(obj2)} and is not supported for add operation.")
    window_obj1 = get_window(obj1)
    window_obj2 = get_window(obj2)
    if window_obj1 is not None and window_obj2 is not None:
        check(window_obj1==window_obj2, TypeError,
              f"For {name} the time window type must match or None but they were {window_obj1} and {window_obj2}.")
        check(obj1.dim[window_obj1] == obj2.dim[window_obj2], ValueError,
              f"For {name} the time window must match or None but they were {window_obj1}={obj1.dim[window_obj1]} and {window_obj2}={obj2.dim[window_obj2]}.")
    check(obj1.dim['dim'] == obj2.dim['dim'] or obj1.dim == {'dim':1} or obj2.dim == {'dim':1}, ValueError,
          f"For {name} the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
    dim = obj1.dim | obj2.dim
    dim['dim'] = max(obj1.dim['dim'], obj2.dim['dim'])
    return obj1, obj2, dim

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
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = aritmetic_cheks(self, obj1, obj2, 'addition operators (+)')
        super().__init__(add_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
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
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = aritmetic_cheks(self, obj1, obj2, 'subtraction operators (-)')
        super().__init__(sub_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
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
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = aritmetic_cheks(self, obj1, obj2, 'multiplication operators (*)')
        super().__init__(mul_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
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
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = aritmetic_cheks(self, obj1, obj2, 'division operators (/) ')
        super().__init__(div_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [div_relation_name,[obj1.name,obj2.name]]

class Pow(Stream, ToStream):
    """
        Implement the power function given an input and an exponent. 
        (it is also possible to use the classical math operator '**')

        See also:
            Official PyTorch pow documentation:
            `torch.pow <https://pytorch.org/docs/stable/generated/torch.pow.html>`_

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
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = aritmetic_cheks(self, obj1, obj2, 'pow operators (**)')
        super().__init__(pow_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [pow_relation_name,[obj1.name,obj2.name]]

class Atan2(Stream, ToStream):
    """
        Implement the atan2 function given an input y and an input x. 

        See also:
            Official PyTorch atan2 documentation:
            `torch.atan2 <https://pytorch.org/docs/stable/generated/torch.atan2.html>`_

        :param y: the first input
        :type obj: Tensor
        :param x: the second input
        :type obj: float or Tensor

        Example:
            >>> atan2 = Atan2(y, x)
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = aritmetic_cheks(self, obj1, obj2, 'atan2 operators (Atan2)')
        super().__init__(atan2_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [atan2_relation_name,[obj1.name,obj2.name]]

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

class Sign(Stream, ToStream):
    """
        Implement the sign function given an input. 

        :param input: the input for the sign function
        :type obj: Tensor

        Example:
            >>> x = Sign(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for sign operation.")
        super().__init__(sign_relation_name+str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [sign_relation_name,[obj.name]]

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

class Atan2_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Atan2_Layer, self).__init__()

    def forward(self, *inputs):
        return torch.atan2(inputs[0], inputs[1])

def createAtan2(name, *inputs):
    #: :noindex:
    return Atan2_Layer()

class Neg_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Neg_Layer, self).__init__()

    def forward(self, x):
        return -x

def createNeg(self, *inputs):
    #: :noindex:
    return Neg_Layer()

class Sign_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Sign_Layer, self).__init__()

    def forward(self, x):
        return torch.sign(x)

def createSign(self, *inputs):
    #: :noindex:
    return Sign_Layer()

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
setattr(Model, atan2_relation_name, createAtan2)

setattr(Model, neg_relation_name, createNeg)
setattr(Model, sign_relation_name, createSign)

setattr(Model, sum_relation_name, createSum)


