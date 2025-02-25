import inspect, copy, textwrap, torch, math

import torch.nn as nn
import numpy as np

from collections.abc import Callable

from nnodely.relation import NeuObj, Stream, toStream
from nnodely.model import Model
from nnodely.parameter import Parameter, Constant
from nnodely.utils import check, merge, enforce_types

from nnodely.linear import Linear
from nnodely.part import Select, Concatenate
from nnodely import *

equationlearner_relation_name = 'EquationLearner'
Available_functions = [Sin, Cos, Tan, Cosh, Sech, Add, Mul, Sub, Neg, Pow, Sum, Concatenate, Relu, Tanh, ELU, Identity, Sigmoid]
Initialized_functions = [ParamFun, Fuzzify]

class EquationLearner(NeuObj):
    """
    Represents a nnodely implementation of the Task-Parametrized Equation Learner block.

    See also:
        Task-Parametrized Equation Learner official paper: 
        `Equation Learner <https://www.sciencedirect.com/science/article/pii/S0921889022001981>`_

    Parameters
    ----------
    functions : list
        A list of callable functions to be used as activation functions.
    linear_in : Linear, optional
        A Linear layer to process the input before applying the activation functions. If not provided a random initialized linear layer will be used instead.
    linear_out : Linear, optional
        A Linear layer to process the output after applying the activation functions. Can be omitted.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    linear_in : Linear or None
        The Linear layer to process the input.
    linear_out : Linear or None
        The Linear layer to process the output.
    functions : list
        The list of activation functions.
    func_parameters : dict
        A dictionary mapping function indices to the number of parameters they require.
    n_activations : int
        The total number of activation functions.

    Examples
    --------

    Example - basic usage:
        >>> x = Input('x')

        >>> equation_learner = EquationLearner(functions=[Tan, Sin, Cos])
        >>> out = Output('out',equation_learner(x.last()))

    Example - passing a linear layer:
        >>> x = Input('x')

        >>> linear_layer = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0})
        >>> equation_learner = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer)

        >>> out = Output('out',equation_learner(x.last()))

    Example - passing a custom parametric function and multiple inputs:
        >>> x = Input('x')
        >>> F = Input('F')

        >>> def myFun(K1,p1):
                return K1*p1

        >>> K = Parameter('k', dimensions =  1, sw = 1,values=[[2.0]])
        >>> parfun = ParamFun(myFun, parameters = [K] )

        >>> equation_learner = EquationLearner([parfun])
        >>> out = Output('out',equation_learner((x.last(),F.last())))
    """
    @enforce_types
    def __init__(self, functions:list, linear_in:Linear|None = None, linear_out:Linear|None = None) -> Stream:

        self.relation_name = equationlearner_relation_name
        self.linear_in = linear_in
        self.linear_out = linear_out

        # input parameters
        self.functions = functions
        super().__init__(equationlearner_relation_name + str(NeuObj.count))

        self.func_parameters = {}
        for func_idx, func in enumerate(self.functions):
            check(callable(func), TypeError, 'The activation functions must be callable')
            if type(func) in Initialized_functions:
                if type(func) == ParamFun:
                    funinfo = inspect.getfullargspec(func.param_fun)
                    num_args = len(funinfo.args) - len(func.parameters) if func.parameters else len(funinfo.args)
                elif type(func) == Fuzzify:
                    init_signature = inspect.signature(func.__call__)  
                    parameters = list(init_signature.parameters.values())
                    num_args = len([param for param in parameters if param.name != "self"])
            else:
                check(func in Available_functions, ValueError, f'The function {func} is not available for the EquationLearner operation')
                init_signature = inspect.signature(func.__init__)  
                parameters = list(init_signature.parameters.values())
                num_args = len([param for param in parameters if param.name != "self"])
            self.func_parameters[func_idx] = num_args

        self.n_activations = sum(self.func_parameters.values())
        check(self.n_activations > 0, ValueError, 'At least one activation function must be provided')

    def __call__(self, inputs):
        if type(inputs) is not tuple:
            inputs = (inputs,)
        check(len(set([x.dim['sw'] if 'sw' in x.dim.keys() else x.dim['tw'] for x in inputs])) == 1, ValueError, 'All inputs must have the same time dimension')
        for input_idx, inp in enumerate(inputs):
            concatenated_input = inp if input_idx == 0 else Concatenate(concatenated_input, inp)
        linear_layer = self.linear_in(concatenated_input) if self.linear_in else Linear(output_dimension=self.n_activations)(concatenated_input)
        idx = 0
        for func_idx, func in enumerate(self.functions):
            arguments = [Select(linear_layer,idx+arg_idx) for arg_idx in range(self.func_parameters[func_idx])]
            idx += self.func_parameters[func_idx]
            out = func(*arguments) if func_idx == 0 else Concatenate(out, func(*arguments))
        if self.linear_out:
            out = self.linear_out(out)
        return out

# class EquationLearner(NeuObj):
#     @enforce_types
#     def __init__(self, functions:list|dict|None = None) -> Stream:

#         self.relation_name = equationlearner_relation_name

#         # input parameters
#         self.functions = functions
#         super().__init__(equationlearner_relation_name + str(NeuObj.count))
#         for func in self.functions:
#             check(callable(func), TypeError, 'The activation functions must be callable')
#             code = textwrap.dedent(inspect.getsource(func)).replace('\"', '\'')
#             self.json['Functions'][self.name] = {
#                 'code' : code,
#                 'name' : func.__name__,
#             }
#         self.json['Functions'][self.name]['params_and_consts'] = []

#     def __call__(self, *obj):
#         stream_name = equationlearner_relation_name + str(Stream.count)

#         for func in self.functions:
#             funinfo = inspect.getfullargspec(func)
#             n_function_input = len(funinfo.args)
#             n_call_input = len(obj)
#             n_new_constants_and_params = n_function_input - n_call_input

#             if 'n_input' not in self.json['Functions'][self.name]:
#                 self.json['Functions'][self.name]['n_input'] = n_call_input
#                 self.__set_params_and_consts(n_new_constants_and_params)

#                 input_dimensions = []
#                 input_types = []
#                 for ind, o in enumerate(obj):
#                     if type(o) in (int,float,list):
#                         obj_type = Constant
#                     else:
#                         obj_type = type(o)
#                     o = toStream(o)
#                     check(type(o) is Stream, TypeError,
#                         f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
#                     input_types.append(obj_type)
#                     input_dimensions.append(o.dim)

#                 self.json['Functions'][self.name]['in_dim'] = copy.deepcopy(input_dimensions)
#                 self.__infer_output_dimensions(input_types, input_dimensions)
#                 self.json['Functions'][self.name]['out_dim'] = copy.deepcopy(self.output_dimension)

#             # Create the missing parameters
#             missing_params = n_new_constants_and_params - len(self.json['Functions'][self.name]['params_and_consts'])
#             check(missing_params == 0, ValueError, f"The function is called with different number of inputs.")

#             stream_json = copy.deepcopy(self.json)
#             input_names = []
#             for ind, o in enumerate(obj):
#                 o = toStream(o)
#                 check(type(o) is Stream, TypeError,
#                     f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
#                 stream_json = merge(stream_json, o.json)
#                 input_names.append(o.name)

#         output_dimension = copy.deepcopy(self.output_dimension)
#         stream_json['Relations'][stream_name] = [equationlearner_relation_name, input_names, self.name]

#         return Stream(stream_name, stream_json, output_dimension)

#     def __set_params_and_consts(self, n_new_constants_and_params):
#         pass


# class EquationLearner_Layer(nn.Module):
#     def __init__(self, functions, input_dim, output_dim, hidden_layers):
#         super(EquationLearner_Layer, self).__init__()
#         self.hidden_layers = nn.ModuleList()

#         # Define activation functions: I (identity), sin, cos, sigma, and sech
#         self.activations = {
#             'identity': lambda x: x,
#             'sin': torch.sin,
#             'cos': torch.cos,
#             'sigma': lambda x: 1 / (1 + torch.exp(-x)),
#             'sech': lambda x: 2 / (torch.exp(x) + torch.exp(-x))
#         }

#         # Input layer
#         current_dim = input_dim
#         for _ in range(hidden_layers):
#             layer = nn.Linear(current_dim, len(functions) * current_dim)
#             self.hidden_layers.append(layer)
#             current_dim = len(functions) * current_dim

#         # Output layer
#         self.output_layer = nn.Linear(current_dim, output_dim)

#     def forward(self, x):
#         for layer in self.hidden_layers:
#             x = layer(x)
#             # Apply all activation functions
#             activation_results = torch.cat([
#                 self.activations["identity"](x),
#                 self.activations["sin"](x),
#                 self.activations["cos"](x),
#                 self.activations["sigma"](x),
#                 self.activations["sech"](x)
#             ], dim=-1)
#             x = activation_results
#         return self.output_layer(x)

# def createEquationLearner(self, *func_params):
#     return EquationLearner_Layer(func=func_params[0])

# setattr(Model, equationlearner_relation_name, createEquationLearner)