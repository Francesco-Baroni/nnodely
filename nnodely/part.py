import copy

import torch.nn as nn
import torch

from nnodely.relation import ToStream, Stream
from nnodely.model import Model
from nnodely.utils import check, enforce_types

part_relation_name = 'Part'
select_relation_name = 'Select'
timepart_relation_name = 'TimePart'
timeselect_relation_name = 'TimeSelect'
samplepart_relation_name = 'SamplePart'
sampleselect_relation_name = 'SampleSelect'

# class Part(Stream, ToStream):
#     @enforce_types
#     def __init__(self, obj:Stream, i:int, j:int):
#         # check(type(obj) is Stream, TypeError,
#         #       f"The type of {obj} is {type(obj)} and is not supported for Part operation.")
#         check(i >= 0 and j > 0 and i < obj.dim['dim'] and j <= obj.dim['dim'],
#               IndexError,
#               f"i={i} or j={j} are not in the range [0,{obj.dim['dim']}]")
#         dim = copy.deepcopy(obj.dim)
#         dim['dim'] = j - i
#         super().__init__(part_relation_name + str(Stream.count),obj.json,dim)
#         self.json['Relations'][self.name] = [part_relation_name,[obj.name],[i,j]]

# class Part_Layer(nn.Module):
#     @enforce_types
#     def __init__(self, i:int, j:int):
#         super(Part_Layer, self).__init__()
#         self.i, self.j = i, j

#     def forward(self, x):
#         assert x.ndim >= 3, 'The Part Relation Works only for 3D inputs'
#         return x[:, :, self.i:self.j]

# ## Select elements on the third dimension in the range [i,j]
# def createPart(self, *inputs):
#     return Part_Layer(i=inputs[0][0], j=inputs[0][1])

class Part(Stream, ToStream):
    """
    Represents a selection of a sub-part from a relation in the neural network model.

    Notes
    -----
    .. note::
        The Part relation works along the object dimension (third dimension) of the input.

    Parameters
    ----------
    obj : Stream
        The stream object to create a part from.
    i : int
        The starting index of the part.
    j : int
        The ending index of the part.

    Attributes
    ----------
    name : str
        The name of the part.
    dim : dict
        A dictionary containing the dimensions of the part.
    json : dict
        A dictionary containing the configuration of the part.

    Example
    -------
        >>> x = Input('x', dimensions=3).last()
        >>> relation = Part(x, 0, 1)

    Raises
    ------
    IndexError
        If the indices i and j are out of range.
    """
    @enforce_types
    def __init__(self, obj:Stream, i:int, j:int):
        # check(type(obj) is Stream, TypeError,
        #       f"The type of {obj} is {type(obj)} and is not supported for Part operation.")
        check(i >= 0 and j > 0 and i < obj.dim['dim'] and j <= obj.dim['dim'],
              IndexError,
              f"i={i} or j={j} are not in the range [0,{obj.dim['dim']}]")
        dim = copy.deepcopy(obj.dim)
        dim['dim'] = j - i
        super().__init__(part_relation_name + str(Stream.count),obj.json,dim)
        self.json['Relations'][self.name] = [part_relation_name,[obj.name],obj.dim['dim'],[i,j]]

class Part_Layer(nn.Module):
    @enforce_types
    def __init__(self, dim:int, i:int, j:int):
        super(Part_Layer, self).__init__()
        self.i, self.j = i, j

        # Create a binary mask matrix for the desired slice
        self.W = torch.zeros((j - i, dim))
        for idx in range(j - i):
            self.W[idx, i + idx] = 1

    def forward(self, x):
        ## assert x.ndim >= 3, 'The Part Relation Works only for 3D inputs'
        return torch.einsum('bij,kj->bik', x, self.W)

## Select elements on the third dimension in the range [i,j]
def createPart(self, *inputs):
    return Part_Layer(dim=inputs[0], i=inputs[1][0], j=inputs[1][1])

# class Select(Stream, ToStream):

#     @enforce_types
#     def __init__(self, obj:Stream, i:int):
#         # check(type(obj) is Stream, TypeError,
#         #       f"The type of {obj} is {type(obj)} and is not supported for Select operation.")
#         check(i >= 0 and i < obj.dim['dim'],
#               IndexError,
#               f"i={i} are not in the range [0,{obj.dim['dim']}]")
#         dim = copy.deepcopy(obj.dim)
#         dim['dim'] = 1
#         super().__init__(select_relation_name + str(Stream.count),obj.json,dim)
#         self.json['Relations'][self.name] = [select_relation_name,[obj.name],i]

# class Select_Layer(nn.Module):
#     def __init__(self, idx):
#         super(Select_Layer, self).__init__()
#         self.idx = idx

#     def forward(self, x):
#         #assert x.ndim >= 3, 'The Part Relation Works only for 3D inputs'
#         return x[:, :, self.idx:self.idx + 1]

# ## Select an element i on the third dimension
# def createSelect(self, *inputs):
#     return Select_Layer(idx=inputs[0])

class Select(Stream, ToStream):
    """
    Represents a selection of a single element from a relation in the neural network model.

    Notes
    -----
    .. note::
        The Select relation works along the object dimension (third dimension) of the input.

    Parameters
    ----------
    obj : Stream
        The stream object to select an element from.
    i : int
        The index of the element to select.

    Attributes
    ----------
    name : str
        The name of the selection.
    dim : dict
        A dictionary containing the dimensions of the selection.
    json : dict
        A dictionary containing the configuration of the selection.

    Example
    -------
        >>> x = Input('x', dimensions=3).last()
        >>> relation = Select(x, 1)

    Raises
    ------
    IndexError
        If the index i is out of range.
    """
    @enforce_types
    def __init__(self, obj:Stream, i:int):
        # check(type(obj) is Stream, TypeError,
        #       f"The type of {obj} is {type(obj)} and is not supported for Select operation.")
        check(i >= 0 and i < obj.dim['dim'],
              IndexError,
              f"i={i} are not in the range [0,{obj.dim['dim']}]")
        dim = copy.deepcopy(obj.dim)
        dim['dim'] = 1
        super().__init__(select_relation_name + str(Stream.count),obj.json,dim)
        self.json['Relations'][self.name] = [select_relation_name,[obj.name],obj.dim['dim'],i]

class Select_Layer(nn.Module):
    def __init__(self, dim, idx):
        super(Select_Layer, self).__init__()
        self.W = torch.zeros(dim)
        self.W[idx] = 1

    def forward(self, x):
        ## assert x.ndim >= 3, 'The Part Relation Works only for 3D inputs'
        return torch.einsum('ijk,k->ij', x, self.W).unsqueeze(2)
    

## Select an element i on the third dimension
def createSelect(self, *inputs):
    return Select_Layer(dim=inputs[0], idx=inputs[1])

class SamplePart(Stream, ToStream):
    """
    Represents a selection of a sub-part from a relation in the neural network model.

    Notes
    -----
    .. note::
        The SamplePart relation works along the time dimension (second dimension) of the input.

    Parameters
    ----------
    obj : Stream
        The stream object to create a part from.
    i : int
        The starting index of the part.
    j : int
        The ending index of the part.
    offset : int, optional
        The offset for the part. Default is None.

    Attributes
    ----------
    name : str
        The name of the part.
    dim : dict
        A dictionary containing the dimensions of the part.
    json : dict
        A dictionary containing the configuration of the part.

    Example
    -------
        >>> x = Input('x').sw(3)
        >>> relation = SamplePart(x, 0, 1)

    Raises
    ------
    KeyError
        If the input does not have a sample window.
    ValueError
        If the indices i and j are out of range or if i is not smaller than j.
    IndexError
        If the offset is not within the sample window.
    """
    @enforce_types
    def __init__(self, obj:Stream, i:int, j:int, offset:int|None = None):
        # check(type(obj) is Stream, TypeError,
        #       f"The type of {obj} is {type(obj)} and is not supported for SamplePart operation.")
        check('sw' in obj.dim, KeyError, 'Input must have a sample window')
        check(i < j, ValueError, 'i must be smaller than j')
        all_inputs = obj.json['Inputs'] | obj.json['States']
        if obj.name in all_inputs:
            backward_idx = all_inputs[obj.name]['sw'][0]
            forward_idx = all_inputs[obj.name]['sw'][1]
        else:
            backward_idx = 0
            forward_idx = obj.dim['sw']
        check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the sample window of the input')
        check(j > backward_idx and j <= forward_idx, ValueError, 'j must be in the sample window of the input')
        dim = copy.deepcopy(obj.dim)
        dim['sw']  = j - i
        super().__init__(samplepart_relation_name + str(Stream.count),obj.json,dim)
        if obj.name in all_inputs:
            rel = [samplepart_relation_name,[obj.name],-1,[i,j]]
        else:
            rel = [samplepart_relation_name,[obj.name],obj.dim['sw'],[i,j]]
        #rel = [samplepart_relation_name,[obj.name],[i,j]]
        if offset is not None:
            check(i <= offset < j, IndexError,"The offset must be inside the sample window")
            rel.append(offset)
        self.json['Relations'][self.name] = rel

# class SamplePart_Layer(nn.Module):
#     def __init__(self, part, offset):
#         super(SamplePart_Layer, self).__init__()
#         self.back, self.forw = part[0], part[1]
#         self.offset = offset

#     def forward(self, x):
#         if self.offset is not None:
#             x = x - x[:, self.offset].unsqueeze(1)
#         result = x[:, self.back:self.forw]
#         return result

class SamplePart_Layer(nn.Module):
    def __init__(self, dim, part, offset):
        super(SamplePart_Layer, self).__init__()
        back, forw = part[0], part[1]
        self.offset = offset
        
        # Create the selection matrix W
        self.W = torch.zeros(forw - back, dim)
        for i in range(forw - back):
            self.W[i, back + i] = 1

    def forward(self, x):
        if self.offset is not None:
            x = x - x[:, self.offset].unsqueeze(1)
        result = torch.einsum('bij,ki->bkj', x, self.W)
        return result

def createSamplePart(self, *inputs):
    if len(inputs) > 2: ## offset
        return SamplePart_Layer(dim=inputs[0], part=inputs[1], offset=inputs[2])
    else:
        return SamplePart_Layer(dim=inputs[0], part=inputs[1], offset=None)

# def createSamplePart(self, *inputs):
#     if len(inputs) > 1: ## offset
#         return SamplePart_Layer(part=inputs[0], offset=inputs[1])
#     else:
#         return SamplePart_Layer(part=inputs[0], offset=None)

# class SampleSelect(Stream, ToStream):

#     @enforce_types
#     def __init__(self, obj:Stream, i:int):
#         # check(type(obj) is Stream, TypeError,
#         #       f"The type of {obj} is {type(obj)} and is not supported for SampleSelect operation.")
#         check('sw' in obj.dim, KeyError, 'Input must have a sample window')
#         backward_idx = 0
#         forward_idx = obj.dim['sw']
#         check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the sample window of the input')
#         dim = copy.deepcopy(obj.dim)
#         del dim['sw']
#         super().__init__(sampleselect_relation_name + str(Stream.count),obj.json,dim)
#         self.json['Relations'][self.name] = [sampleselect_relation_name,[obj.name],i]

# class SampleSelect_Layer(nn.Module):
#     def __init__(self, idx):
#         super(SampleSelect_Layer, self).__init__()
#         self.idx = idx

#     def forward(self, x):
#         #assert x.ndim >= 2, 'The Part Relation Works only for 2D inputs'
#         return x[:, self.idx:self.idx + 1, :]

# def createSampleSelect(self, *inputs):
#     return SampleSelect_Layer(idx=inputs[0])

class SampleSelect(Stream, ToStream):
    """
    Represents a selection of a single element from a relation in the neural network model.

    Notes
    -----
    .. note::
        The SampleSelect relation works along the time dimension (second dimension) of the input.

    Parameters
    ----------
    obj : Stream
        The stream object to select an element from.
    i : int
        The index of the element to select.

    Attributes
    ----------
    name : str
        The name of the selection.
    dim : dict
        A dictionary containing the dimensions of the selection.
    json : dict
        A dictionary containing the configuration of the selection.

    Example
    -------
        >>> x = Input('x').sw(3)
        >>> relation = SampleSelect(x, 1)

    Raises
    ------
    IndexError
        If the index i is out of range.
    KeyError
        If the input does not have a sample window.
    IndexError
        If the offset is not within the sample window.
    """
    @enforce_types
    def __init__(self, obj:Stream, i:int):
        # check(type(obj) is Stream, TypeError,
        #       f"The type of {obj} is {type(obj)} and is not supported for SampleSelect operation.")
        check('sw' in obj.dim, KeyError, 'Input must have a sample window')
        backward_idx = 0
        forward_idx = obj.dim['sw']
        check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the sample window of the input')
        dim = copy.deepcopy(obj.dim)
        del dim['sw']
        super().__init__(sampleselect_relation_name + str(Stream.count),obj.json,dim)
        self.json['Relations'][self.name] = [sampleselect_relation_name,[obj.name],obj.dim['sw'],i]

class SampleSelect_Layer(nn.Module):
    def __init__(self, dim, idx):
        super(SampleSelect_Layer, self).__init__()
        self.W = torch.zeros(dim)
        self.W[idx] = 1

    def forward(self, x):
        return torch.einsum('ijk,j->ik', x, self.W).unsqueeze(1)

def createSampleSelect(self, *inputs):
    return SampleSelect_Layer(dim=inputs[0], idx=inputs[1])

class TimePart(Stream, ToStream):
    """
    Represents a part of a stream in the neural network model along the time dimension (second dimension).

    Parameters
    ----------
    obj : Stream
        The stream object to create a part from.
    i : int or float
        The starting index of the part.
    j : int or float
        The ending index of the part.
    offset : int or float, optional
        The offset for the part. Default is None.

    Attributes
    ----------
    name : str
        The name of the part.
    dim : dict
        A dictionary containing the dimensions of the part.
    json : dict
        A dictionary containing the configuration of the part.

    Example
    -------
        >>> x = Input('x').sw(10)
        >>> time_part = TimePart(x, i=0, j=5)

    Raises
    ------
    KeyError
        If the input does not have a time window.
    ValueError
        If the indices i and j are out of range or if i is not smaller than j.
    IndexError
        If the offset is not within the time window.
    """
    @enforce_types
    def __init__(self, obj:Stream, i:int|float, j:int|float, offset:int|float|None = None):
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for TimePart operation.")
        check('tw' in obj.dim, KeyError, 'Input must have a time window')
        check(i < j, ValueError, 'i must be smaller than j')
        all_inputs = obj.json['Inputs'] | obj.json['States']
        if obj.name in all_inputs:
            backward_idx = all_inputs[obj.name]['tw'][0]
            forward_idx = all_inputs[obj.name]['tw'][1]
        else:
            backward_idx = 0
            forward_idx = obj.dim['tw']
        check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the time window of the input')
        check(j > backward_idx and j <= forward_idx, ValueError, 'j must be in the time window of the input')
        dim = copy.deepcopy(obj.dim)
        dim['tw']  = j - i
        super().__init__(timepart_relation_name + str(Stream.count),obj.json,dim)
        if obj.name in all_inputs:
            rel = [timepart_relation_name,[obj.name],-1,[i,j]]
        else:
            rel = [timepart_relation_name,[obj.name],obj.dim['tw'],[i,j]]
        #rel = [timepart_relation_name,[obj.name],[i,j]]
        if offset is not None:
            check(i <= offset < j, IndexError,"The offset must be inside the time window")
            rel.append(offset)
        self.json['Relations'][self.name] = rel

# class TimePart_Layer(nn.Module):
#     def __init__(self, part, offset):
#         super(TimePart_Layer, self).__init__()
#         self.back, self.forw = part[0], part[1]
#         self.offset = offset

#     def forward(self, x):
#         if self.offset is not None:
#             x = x - x[:, self.offset].unsqueeze(1)
#         return x[:, self.back:self.forw]

class TimePart_Layer(nn.Module):
    def __init__(self, dim, part, offset):
        super(TimePart_Layer, self).__init__()
        back, forw = part[0], part[1]
        self.offset = offset

        # Create the selection matrix W
        self.W = torch.zeros(size=(forw - back, int(dim)))
        for i in range(forw - back):
            self.W[i, back + i] = 1

    def forward(self, x):
        if self.offset is not None:
            x = x - x[:, self.offset].unsqueeze(1)
        result = torch.einsum('bij,ki->bkj', x, self.W)
        return result

def createTimePart(self, *inputs):
    if len(inputs) > 2: ## offset
        return TimePart_Layer(dim=inputs[0], part=inputs[1], offset=inputs[2])
    else:
        return TimePart_Layer(dim=inputs[0], part=inputs[1], offset=None)

# def createTimePart(self, *inputs):
#     if len(inputs) > 1: ## offset
#         return TimePart_Layer(part=inputs[0], offset=inputs[1])
#     else:
#         return TimePart_Layer(part=inputs[0], offset=None)

class TimeSelect(Stream, ToStream):

    @enforce_types
    def __init__(self, obj:Stream, i:int|float):
        check('tw' in obj.dim, KeyError, 'Input must have a time window')
        backward_idx = 0
        forward_idx = obj.dim['tw']
        check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the time window of the input')
        dim = copy.deepcopy(obj.dim)
        del dim['tw']
        super().__init__(timeselect_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = [timeselect_relation_name,[obj.name],i]

setattr(Model, part_relation_name, createPart)
setattr(Model, select_relation_name, createSelect)

setattr(Model, samplepart_relation_name, createSamplePart)
setattr(Model, sampleselect_relation_name, createSampleSelect)

setattr(Model, timepart_relation_name, createTimePart)
