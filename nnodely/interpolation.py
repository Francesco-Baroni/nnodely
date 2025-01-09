import copy, inspect, textwrap, torch

import torch.nn as nn

from collections.abc import Callable

from nnodely.relation import NeuObj, Stream, AutoToStream
from nnodely.model import Model
from nnodely.utils import check, merge, enforce_types

from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

interpolation_relation_name = 'Interpolation'
class Interpolation(NeuObj, AutoToStream):
    """
    Represents a Linear relation in the neural network model.

    Notes
    -----
    .. note::
        The Linear relation works along the input dimension (third dimension) of the input tensor.
        You can find some initialization functions inside the initializer module.

    Parameters
    ----------
    output_dimension : int, optional
        The output dimension of the Linear relation.
    W_init : Callable, optional
        A callable for initializing the weights.
    W_init_params : dict, optional
        A dictionary of parameters for the weight initializer.
    b_init : Callable, optional
        A callable for initializing the bias.
    b_init_params : dict, optional
        A dictionary of parameters for the bias initializer.
    W : Parameter or str, optional
        The weight parameter object or name. If not given a new parameter will be auto-generated.
    b : bool, str, or Parameter, optional
        The bias parameter object, name, or a boolean indicating whether to use bias. If set to 'True' a new parameter will be auto-generated.
    dropout : int or float, optional
        The dropout rate. Default is 0.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    W_init : Callable
        The weight initializer.
    W_init_params : dict
        The parameters for the weight initializer.
    b_init : Callable
        The bias initializer.
    b_init_params : dict
        The parameters for the bias initializer.
    W : Parameter or str
        The weight parameter object or name.
    b : bool, str, or Parameter
        The bias parameter object, name, or a boolean indicating whether to use bias.
    Wname : str
        The name of the weight parameter.
    bname : str
        The name of the bias parameter.
    dropout : int or float
        The dropout rate.
    output_dimension : int
        The output dimension of the Linear relation.

    Examples
    --------

    Example - basic usage:
        >>> input = Input('in').tw(0.05)
        >>> relation = Linear(input)

    Example - passing a weight and bias parameter:
        >>> input = Input('in').last()
        >>> weight = Parameter('W', values=[[[1]]])
        >>> bias = Parameter('b', values=[[1]])
        >>> relation = Linear(W=weight, b=bias)(input)

    Example - parameters initialization:
        >>> input = Input('in').last()
        >>> relation = Linear(b=True, W_init=init_negexp, b_init=init_constant, b_init_params={'value':1})(input)
    """

    @enforce_types
    def __init__(self, x_points:list|None = None,
                 y_points:list|None = None,
                 mode:str|None = 'linear'):

        self.relation_name = interpolation_relation_name
        self.x_points = x_points
        self.y_points = y_points
        self.mode = mode

        self.available_modes = ['linear', 'polynomial']

        super().__init__('P' + interpolation_relation_name + str(NeuObj.count))
        check(len(x_points) == len(y_points), ValueError, 'The x_points and y_points must have the same length.')
        check(mode in self.available_modes, ValueError, f'The mode must be one of {self.available_modes}.')

    def __call__(self, obj:Stream) -> Stream:
        stream_name = interpolation_relation_name + str(Stream.count)
        check(type(obj) is Stream, TypeError, f"The type of {obj} is {type(obj)} and is not supported for Interpolation operation.")

        stream_json = merge(self.json,obj.json)
        stream_json['Relations'][stream_name] = [interpolation_relation_name, [obj.name], self.x_points, self.y_points, self.mode]
        return Stream(stream_name, stream_json, obj.dim)


class Interpolation_Layer(nn.Module):
    def __init__(self, x_points, y_points, mode='linear'):
        super(Interpolation_Layer, self).__init__()
        self.mode = mode
        ## Sort the points
        if type(x_points) is not torch.Tensor:
            x_points = torch.tensor(x_points)
        if type(y_points) is not torch.Tensor:
            y_points = torch.tensor(y_points)
        self.x_points, indices = torch.sort(x_points)
        self.y_points = y_points[indices]

    def forward(self, x):
        if self.mode == 'linear':
            return self.linear_interpolation(x)
        else:
            raise NotImplementedError
    
    def linear_interpolation(self, x):
        x_interpolated = torch.zeros_like(x)
        for i, val in enumerate(x):
            # Find the interval [x1, x2] such that x1 <= val <= x2
            idx = torch.searchsorted(self.x_points, val).item()
            if idx == 0:
                # val is less than the smallest x_point, extrapolate
                x_interpolated[i] = self.y_points[0]
            elif idx >= len(self.x_points):
                # val is greater than the largest x_point, extrapolate
                x_interpolated[i] = self.y_points[-1]
            else:
                # Perform linear interpolation between x_points[idx-1] and x_points[idx]
                x1, x2 = self.x_points[idx - 1], self.x_points[idx]
                y1, y2 = self.y_points[idx - 1], self.y_points[idx]
                # Linear interpolation formula
                x_interpolated[i] = y1 + (val - x1) * (y2 - y1) / (x2 - x1)
        return x_interpolated

def createInterpolation(self, *inputs):
    return Interpolation_Layer(x_points=inputs[0], y_points=inputs[1], mode=inputs[2])

setattr(Model, interpolation_relation_name, createInterpolation)