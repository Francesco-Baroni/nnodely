import copy, inspect, textwrap
import numpy as np

from collections.abc import Callable

from nnodely.relation import NeuObj, Stream
from nnodely.utils import check, enforce_types, NP_DTYPE

def is_numpy_float(var):
    return isinstance(var, (np.float16, np.float32, np.float64))

class Constant(NeuObj, Stream):
    """
    Represents a constant value in the neural network model.

    Parameters
    ----------
    name : str
        The name of the constant.
    values : list, float, int, or np.ndarray
        The values of the constant.
    tw : float or int, optional
        The time window for the constant. Default is None.
    sw : int, optional
        The sample window for the constant. Default is None.

    Attributes
    ----------
    name : str
        The name of the constant.
    dim : dict
        A dictionary containing the dimensions of the constant.
    json : dict
        A dictionary containing the configuration of the constant.

    Example
    -------
        >>> g = Constant('gravity',values=9.81)
    """
    @enforce_types
    def __init__(self, name:str,
                 values:list|float|int|np.ndarray,
                 tw:float|int|None = None,
                 sw:int|None = None):

        NeuObj.__init__(self, name)
        values = np.array(values, dtype=NP_DTYPE)
        shape = values.shape
        values = values.tolist()
        if len(shape) == 0:
            self.dim = {'dim': 1}
        else:
            check(len(shape) >= 2, ValueError,
              f"The shape of a Constant must have at least 2 dimensions or zero.")
            dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
            self.dim = {'dim': dimensions}
            if tw is not None:
                check(sw is None, ValueError, "If tw is set sw must be None")
                self.dim['tw'] = tw
            elif sw is not None:
                self.dim['sw'] = sw
                check(shape[0] == self.dim['sw'],ValueError, f"The sw = {sw} is different from sw = {shape[0]} of the values.")
            else:
                self.dim['sw'] = shape[0]

        # deepcopy dimention information inside Parameters
        self.json['Constants'][self.name] = copy.deepcopy(self.dim)
        self.json['Constants'][self.name]['values'] = values
        Stream.__init__(self, name, self.json, self.dim)

class Parameter(NeuObj, Stream):
    """
    Represents a parameter in the neural network model.

    Notes
    -----
    .. note::
        You can find some initialization functions for the 'init' and 'init_params' parameters inside the initializer module.

    Parameters
    ----------
    name : str
        The name of the parameter.
    dimensions : int, list, tuple, or None, optional
        The dimensions of the parameter. Default is None.
    tw : float or int, optional
        The time window for the parameter. Default is None.
    sw : int, optional
        The sample window for the parameter. Default is None.
    values : list, float, int, np.ndarray, or None, optional
        The values of the parameter. Default is None.
    init : Callable, optional
        A callable for initializing the parameter values. Default is None.
    init_params : dict, optional
        A dictionary of parameters for the initializer. Default is None.

    Attributes
    ----------
    name : str
        The name of the parameter.
    dim : dict
        A dictionary containing the dimensions of the parameter.
    json : dict
        A dictionary containing the configuration of the parameter.

    Examples
    --------

    Example - basic usage:
        >>> k = Parameter('k', dimensions=3, tw=4)

    Example - initialize a parameter with values:
        >>> x = Input('x')
        >>> gravity = Parameter('g', dimensions=(4,1),values=[[[1],[2],[3],[4]]])
        >>> out = Output('out', Linear(W=gravity)(x.sw(3)))

    Example - initialize a parameter with a function:
        >>> x = Input('x').last()
        >>> p = Parameter('param', dimensions=1, sw=1, init=init_constant, init_params={'value':1})
        >>> relation = Fir(parameter=param)(x)
    """
    @enforce_types
    def __init__(self, name:str,
                 dimensions:int|list|tuple|None = None,
                 tw:float|int|None = None,
                 sw:int|None = None,
                 values:list|float|int|np.ndarray|None = None,
                 init:Callable|None = None,
                 init_params:dict|None = None):

        NeuObj.__init__(self, name)
        dimensions = list(dimensions) if type(dimensions) is tuple else dimensions
        if values is None:
            if dimensions is None:
                dimensions = 1
            self.dim = {'dim': dimensions}
            if tw is not None:
                check(sw is None, ValueError, "If tw is set sw must be None")
                self.dim['tw'] = tw
            elif sw is not None:
                self.dim['sw'] = sw

            # deepcopy dimention information inside Parameters
            self.json['Parameters'][self.name] = copy.deepcopy(self.dim)
        else:
            values = np.array(values, dtype=NP_DTYPE)
            shape = values.shape
            values = values.tolist()
            check(len(shape) >= 2, ValueError,
                  f"The shape of a parameter must have at least 2 dimensions.")
            values_dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
            if dimensions is None:
                dimensions = values_dimensions
            else:
                check(dimensions == values_dimensions, ValueError,
                      f"The dimensions = {dimensions} are different from dimensions = {values_dimensions} of the values.")
            self.dim = {'dim': dimensions}

            if tw is not None:
                check(sw is None, ValueError, "If tw is set sw must be None")
                self.dim['tw'] = tw
            elif sw is not None:
                self.dim['sw'] = sw
                check(shape[0] == self.dim['sw'],ValueError, f"The sw = {sw} is different from sw = {shape[0]} of the values.")
            else:
                self.dim['sw'] = shape[0]

            # deepcopy dimention information inside Parameters
            self.json['Parameters'][self.name] = copy.deepcopy(self.dim)
            self.json['Parameters'][self.name]['values'] = values

        if init is not None:
            check('values' not in self.json['Parameters'][self.name], ValueError, f"The parameter {self.name} is already initialized.")
            check(inspect.isfunction(init), ValueError,f"The init parameter must be a function.")
            code = textwrap.dedent(inspect.getsource(init)).replace('\"', '\'')
            self.json['Parameters'][self.name]['init_fun'] = { 'code' : code, 'name' : init.__name__}
            if init_params is not None:
                self.json['Parameters'][self.name]['init_fun']['params'] = init_params

        Stream.__init__(self, name, self.json, self.dim)
