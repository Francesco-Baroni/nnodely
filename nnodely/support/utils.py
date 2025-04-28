import copy, torch, inspect, typing
import types

from collections import OrderedDict

import numpy as np
from contextlib import suppress
from pprint import pformat
from functools import wraps
from typing import get_type_hints
import keyword

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

TORCH_DTYPE = torch.float32
NP_DTYPE = np.float32

ForbiddenTags = keyword.kwlist

class ReadOnlyDict:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        value = self._data[key]
        if isinstance(value, dict):
            return dict(ReadOnlyDict(value))
        return value

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def __repr__(self):
        from pprint import pformat
        return pformat(self._data)

    def __or__(self, other):
        if not isinstance(other, ReadOnlyDict):
            return NotImplemented
        combined_data = {**self._data, **other._data}
        return ReadOnlyDict(combined_data)

    def __str__(self):
        from nnodely.visualizer.emptyvisualizer import color, GREEN
        from pprint import pformat
        return color(pformat(self._data), GREEN)

    def __eq__(self, other):
        if not isinstance(other, ReadOnlyDict):
            return self._data == other
        return self._data == other._data

    # def to_dict(self):
    #     # Convert the ReadOnlyDict to a regular dictionary
    #     return {key: value.to_dict() if isinstance(value, ReadOnlyDict) else value for key, value in self._data.items()}


def get_window(obj):
    return 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)

# def get_inputs(json, relation):
#     # Get all the inputs needed to compute a specific relation from the json graph
#     inputs = []
#
#     def search(rel):
#         if rel in (json['Inputs'] | json['States']):  # Found an input
#             inputs.append(rel)
#             if rel in json['States']:
#                 if 'connect' in json['States'][rel]:
#                     search(json['States'][rel]['connect'])
#                 if 'closed_loop' in json['States'][rel]:
#                     search(json['States'][rel]['closed_loop'])
#         elif rel in json['Relations']:  # Another relation
#             for sub_rel in json['Relations'][rel][1]:
#                 search(sub_rel)
#
#     for rel in json['Relations'][relation][1]:
#         search(rel)
#
#     return inputs

# def subjson(json, models:str|list):
#     from nnodely.basic.relation import MAIN_JSON
#     if type(models) is not str:
#         sub_json = {}
#         for model in models:
#             check(model in json['Models'],AttributeError, f"Model [{model}] not found!")
#             for category, items in json['Models'][model].items():
#                 sub_json[category] = {key:value for key, value in json[category].items() if key in items}
#             sub_json['Models'] = {}
#             for model in models:
#                 sub_json['Models'][model] = {key:value for key,value in json['Models'][model].items()}
#     else:
#         sub_json = copy.deepcopy(json)
#         sub_json['Minimizers'] = {}
#         sub_json['Info'] = {}
#     return sub_json

def subjson_from_relation(json, relation):
    # Get all the inputs needed to compute a specific relation from the json graph
    inputs = set()
    relations = set()
    constants = set()
    parameters = set()
    functions = set()

    def search(rel):
        if rel in json['Inputs']:  # Found an input
            inputs.add(rel)
            if rel in json['Inputs']:
                if 'connect' in json['Inputs'][rel]:
                    search(json['Inputs'][rel]['connect'])
                if 'closed_loop' in json['Inputs'][rel]:
                    search(json['Inputs'][rel]['closed_loop'])
                # if 'init' in json['Inputs'][rel]:
                #     search(json['Inputs'][rel]['init'])
        elif rel in json['Constants']:  # Found a constant or parameter
            constants.add(rel)
        elif rel in json['Parameters']:
            parameters.add(rel)
        elif rel in json['Functions']:
            functions.add(rel)
            if 'params_and_consts' in json['Functions'][rel]:
                for sub_rel in json['Functions'][rel]['params_and_consts']:
                    search(sub_rel)
        elif rel in json['Relations']:  # Another relation
            relations.add(rel)
            for sub_rel in json['Relations'][rel][1]:
                search(sub_rel)
            for sub_rel in json['Relations'][rel][2:]:
                if json['Relations'][rel][0] in ('Fir', 'Linear'):
                    search(sub_rel)
                if json['Relations'][rel][0] in ('Fuzzify'):
                    search(sub_rel)
                if json['Relations'][rel][0] in ('ParamFun'):
                    search(sub_rel)

    search(relation)
    from nnodely.basic.relation import MAIN_JSON
    sub_json = copy.deepcopy(MAIN_JSON)
    sub_json['Relations'] = {key: value for key, value in json['Relations'].items() if key in relations}
    sub_json['Inputs'] = {key: value for key, value in json['Inputs'].items() if key in inputs}
    sub_json['Constants'] = {key: value for key, value in json['Constants'].items() if key in constants}
    sub_json['Parameters'] = {key: value for key, value in json['Parameters'].items() if key in parameters}
    sub_json['Functions'] = {key: value for key, value in json['Functions'].items() if key in functions}
    sub_json['Outputs'] = {}
    sub_json['Minimizers'] = {}
    sub_json['Info'] = {}
    return sub_json


def subjson_from_output(json, outputs:str|list):
    from nnodely.basic.relation import MAIN_JSON
    sub_json = copy.deepcopy(MAIN_JSON)
    if type(outputs) is str:
        outputs = [outputs]
    for output in outputs:
        sub_json = merge(sub_json, subjson_from_relation(json,json['Outputs'][output]))
        sub_json['Outputs'][output] = json['Outputs'][output]
    return sub_json

def subjson_from_model(json, models:str|list):
    from nnodely.basic.relation import MAIN_JSON
    sub_json = copy.deepcopy(MAIN_JSON)
    if type(models) is str:
        check(models in json['Models'], AttributeError, f"Model [{models}] not found!")
        if type(json['Models']) is str:
            outputs = set(json['Outputs'].keys())
        else:
            outputs = set(json['Models'][models]['Outputs'])
        sub_json['Models'] = models
    else:
        outputs = set()
        sub_json['Models'] = {}
        for model in models:
            check(model in json['Models'].keys(), AttributeError, f"Model [{model}] not found!")
            outputs |= set(json['Models'][model]['Outputs'])
            sub_json['Models'][model] = {key: value for key, value in json['Models'][model].items()}

    return merge(sub_json, subjson_from_output(json, outputs))

def enforce_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        all_args = kwargs.copy()

        sig = OrderedDict(inspect.signature(func).parameters)
        if len(sig) != len(args):
            var_type = None
            for ind, arg in enumerate(args):
                if ind < len(list(sig.values())) and list(sig.values())[ind].kind == inspect.Parameter.VAR_POSITIONAL:
                    var_name = list(sig.keys())[ind]
                    var_type = sig.pop(var_name)
                if var_type:
                    sig[var_name+str(ind)] = var_type

        all_args.update(dict(zip(sig, args)))
        if 'self' in sig.keys():
            sig.pop('self')

        for arg_name, arg in all_args.items():
            if (arg_name in hints.keys() or arg_name in sig.keys()) and not isinstance(arg,sig[arg_name].annotation):
                class_name = func.__qualname__.split('.')[0]
                if isinstance(sig[arg_name].annotation, types.UnionType):
                    type_list = [val.__name__ for val in sig[arg_name].annotation.__args__]
                else:
                    type_list = sig[arg_name].annotation.__name__
                raise TypeError(
                    f"In Function or Class {class_name} the argument '{arg_name}' to be of type {type_list}, but got {type(arg).__name__}")

        # for arg, arg_type in hints.items():
        #     if arg in all_args and not isinstance(all_args[arg], arg_type):
        #         raise TypeError(
        #             f"In Function or Class {func} Expected argument '{arg}' to be of type {arg_type}, but got {type(all_args[arg]).__name__}")

        return func(*args, **kwargs)

    return wrapper


# Linear interpolation function, operating on batches of input data and returning batches of output data
def linear_interp(x,x_data,y_data):
    # Inputs: 
    # x: query point, a tensor of shape torch.Size([N, 1, 1])
    # x_data: map of x values, sorted in ascending order, a tensor of shape torch.Size([Q, 1])
    # y_data: map of y values, a tensor of shape torch.Size([Q, 1])
    # Output:
    # y: interpolated value at x, a tensor of shape torch.Size([N, 1, 1])

    # Saturate x to the range of x_data
    x = torch.min(torch.max(x,x_data[0]),x_data[-1])

    # Find the index of the closest value in x_data
    idx = torch.argmin(torch.abs(x_data[:-1] - x),dim=1)
    
    # Linear interpolation
    y = y_data[idx] + (y_data[idx+1] - y_data[idx])/(x_data[idx+1] - x_data[idx])*(x - x_data[idx])
    return y

def tensor_to_list(data):
    if isinstance(data, torch.Tensor):
        # Converte il tensore in una lista
        return data.tolist()
    elif isinstance(data, dict):
        # Ricorsione per i dizionari
        return {key: tensor_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Ricorsione per le liste
        return [tensor_to_list(item) for item in data]
    elif isinstance(data, tuple):
        # Ricorsione per tuple
        return tuple(tensor_to_list(item) for item in data)
    elif isinstance(data, torch.nn.modules.container.ParameterDict):
        # Ricorsione per parameter dict
        return {key: tensor_to_list(value) for key, value in data.items()}
    else:
        # Altri tipi di dati rimangono invariati
        return data

# Codice per comprimere le relazioni
        #print(self.json['Relations'])
        # used_rel = {string for values in self.json['Relations'].values() for string in values[1]}
        # if obj1.name not in used_rel and obj1.name in self.json['Relations'].keys() and self.json['Relations'][obj1.name][0] == add_relation_name:
        #     self.json['Relations'][self.name] = [add_relation_name, self.json['Relations'][obj1.name][1]+[obj2.name]]
        #     del self.json['Relations'][obj1.name]
        # else:
        # Devo aggiungere un operazione che rimuove un operazione di Add,Sub,Mul,Div se può essere unita ad un'altra operazione dello stesso tipo
        #
def merge(source, destination, main = True):
    if main:
        for key, value in destination["Functions"].items():
            if key in source["Functions"].keys() and 'n_input' in value.keys() and 'n_input' in source["Functions"][key].keys():
                check(value == {} or source["Functions"][key] == {} or value['n_input'] == source["Functions"][key]['n_input'],
                      TypeError,
                      f"The ParamFun {key} is present multiple times, with different number of inputs. "
                      f"The ParamFun {key} is called with {value['n_input']} parameters and with {source['Functions'][key]['n_input']} parameters.")
        for key, value in destination["Parameters"].items():
            if key in source["Parameters"].keys():
                if 'dim' in value.keys() and 'dim' in source["Parameters"][key].keys():
                    check(value['dim'] == source["Parameters"][key]['dim'],
                          TypeError,
                          f"The Parameter {key} is present multiple times, with different dimensions. "
                          f"The Parameter {key} is called with {value['dim']} dimension and with {source['Parameters'][key]['dim']} dimension.")
                window_dest = 'tw' if 'tw' in value else ('sw' if 'sw' in value else None)
                window_source = 'tw' if 'tw' in source["Parameters"][key] else ('sw' if 'sw' in source["Parameters"][key] else None)
                if window_dest is not None:
                    check(window_dest == window_source and value[window_dest] == source["Parameters"][key][window_source] ,
                          TypeError,
                          f"The Parameter {key} is present multiple times, with different window. "
                          f"The Parameter {key} is called with {window_dest}={value[window_dest]} dimension and with {window_source}={source['Parameters'][key][window_source]} dimension.")

        log.debug("Merge Source")
        log.debug("\n"+pformat(source))
        log.debug("Merge Destination")
        log.debug("\n"+pformat(destination))
        result = copy.deepcopy(destination)
    else:
        result = destination
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = result.setdefault(key, {})
            merge(value, node, False)
        else:
            if key in result and type(result[key]) is list:
                if key == 'tw' or key == 'sw':
                    if result[key][0] > value[0]:
                        result[key][0] = value[0]
                    if result[key][1] < value[1]:
                        result[key][1] = value[1]
            else:
                result[key] = value
    if main == True:
        log.debug("Merge Result")
        log.debug("\n" + pformat(result))
    return result

def check(condition, exception, string):
    if not condition:
        raise exception(string)

def argmax_max(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])

def argmin_min(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])

def argmax_dict(iterable: dict):
    return max(iterable.items(), key=lambda x: x[1])

def argmin_dict(iterable: dict):
    return min(iterable.items(), key=lambda x: x[1])

# Function used to verified the number of gradient operations in the graph
# def count_gradient_operations(grad_fn):
#     count = 0
#     if grad_fn is None:
#         return count
#     nodes = [grad_fn]
#     while nodes:
#         node = nodes.pop()
#         count += 1
#         nodes.extend(next_fn[0] for next_fn in node.next_functions if next_fn[0] is not None)
#     return count

# def check_gradient_operations(X:dict):
#     count = 0
#     for key in X.keys():
#         count += count_gradient_operations(X[key].grad_fn)
#     return count