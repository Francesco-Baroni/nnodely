import inspect, copy, textwrap, torch, math

import torch.nn as nn
import numpy as np

from collections.abc import Callable

from nnodely.relation import NeuObj, Stream, toStream
from nnodely.model import Model
from nnodely.parameter import Parameter, Constant
from nnodely.utils import check, merge, enforce_types


equationlearner_relation_name = 'EquationLearner'

class EquationLearner(NeuObj):
    @enforce_types
    def __init__(self, functions:list|dict|None = None) -> Stream:

        self.relation_name = equationlearner_relation_name

        # input parameters
        self.functions = functions
        super().__init__(equationlearner_relation_name + str(NeuObj.count))
        for func in self.functions:
            code = textwrap.dedent(inspect.getsource(func)).replace('\"', '\'')
            self.json['Functions'][self.name] = {
                'code' : code,
                'name' : func.__name__,
            }
        self.json['Functions'][self.name]['params_and_consts'] = []

    def __call__(self, *obj):
        stream_name = equationlearner_relation_name + str(Stream.count)

        for func in self.functions:
            funinfo = inspect.getfullargspec(func)
            n_function_input = len(funinfo.args)
            n_call_input = len(obj)
            n_new_constants_and_params = n_function_input - n_call_input

            if 'n_input' not in self.json['Functions'][self.name]:
                self.json['Functions'][self.name]['n_input'] = n_call_input
                self.__set_params_and_consts(n_new_constants_and_params)

                input_dimensions = []
                input_types = []
                for ind, o in enumerate(obj):
                    if type(o) in (int,float,list):
                        obj_type = Constant
                    else:
                        obj_type = type(o)
                    o = toStream(o)
                    check(type(o) is Stream, TypeError,
                        f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
                    input_types.append(obj_type)
                    input_dimensions.append(o.dim)

                self.json['Functions'][self.name]['in_dim'] = copy.deepcopy(input_dimensions)
                self.__infer_output_dimensions(input_types, input_dimensions)
                self.json['Functions'][self.name]['out_dim'] = copy.deepcopy(self.output_dimension)

            # Create the missing parameters
            missing_params = n_new_constants_and_params - len(self.json['Functions'][self.name]['params_and_consts'])
            check(missing_params == 0, ValueError, f"The function is called with different number of inputs.")

            stream_json = copy.deepcopy(self.json)
            input_names = []
            for ind, o in enumerate(obj):
                o = toStream(o)
                check(type(o) is Stream, TypeError,
                    f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
                stream_json = merge(stream_json, o.json)
                input_names.append(o.name)

        output_dimension = copy.deepcopy(self.output_dimension)
        stream_json['Relations'][stream_name] = [equationlearner_relation_name, input_names, self.name]

        return Stream(stream_name, stream_json, output_dimension)

    def __set_params_and_consts(self, n_new_constants_and_params):
        pass


class EquationLearner_Layer(nn.Module):
    def __init__(self, functions, input_dim, output_dim, hidden_layers):
        super(EquationLearner_Layer, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # Define activation functions: I (identity), sin, cos, sigma, and sech
        self.activations = {
            'identity': lambda x: x,
            'sin': torch.sin,
            'cos': torch.cos,
            'sigma': lambda x: 1 / (1 + torch.exp(-x)),
            'sech': lambda x: 2 / (torch.exp(x) + torch.exp(-x))
        }

        # Input layer
        current_dim = input_dim
        for _ in range(hidden_layers):
            layer = nn.Linear(current_dim, len(functions) * current_dim)
            self.hidden_layers.append(layer)
            current_dim = len(functions) * current_dim

        # Output layer
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            # Apply all activation functions
            activation_results = torch.cat([
                self.activations["identity"](x),
                self.activations["sin"](x),
                self.activations["cos"](x),
                self.activations["sigma"](x),
                self.activations["sech"](x)
            ], dim=-1)
            x = activation_results
        return self.output_layer(x)

def createEquationLearner(self, *func_params):
    return EquationLearner_Layer(func=func_params[0])

setattr(Model, equationlearner_relation_name, createEquationLearner)