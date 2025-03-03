import inspect, copy, textwrap, torch, math

import torch.nn as nn
import numpy as np

from typing import Union
from collections.abc import Callable

from nnodely.relation import NeuObj, Stream, toStream
from nnodely.model import Model
from nnodely.parameter import Parameter, Constant
from nnodely.utils import check, merge, enforce_types

from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)


paramfun_relation_name = 'ParamFun'

class ParamFun(NeuObj):
    """
    Represents a parametric function in the neural network model.

    Parameters
    ----------
    param_fun : Callable
        The parametric function to be used.
    constants : list or dict or None, optional
        A list or dictionary of constants to be used in the function. Default is None.
    parameters_dimensions : list or dict or None, optional
        A list or dictionary specifying the dimensions of the parameters. Default is None.
    parameters : list or dict or None, optional
        A list or dictionary of parameters to be used in the function. Default is None.
    map_over_batch : bool, optional
        A boolean indicating whether to map the function over the batch dimension. Default is False.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    param_fun : Callable
        The parametric function to be used.
    constants : list or dict or None
        A list or dictionary of constants to be used in the function.
    parameters_dimensions : list or dict or None
        A list or dictionary specifying the dimensions of the parameters.
    parameters : list or dict or None
        A list or dictionary of parameters to be used in the function.
    map_over_batch : bool
        A boolean indicating whether to map the function over the batch dimension.
    output_dimension : dict
        A dictionary containing the output dimensions of the function.
    json : dict
        A dictionary containing the configuration of the function.

    Example
    -------
        >>> input1 = Input('input1')
        >>> input2 = Input('input2')

        >>> def my_function(x, y, param1, const1):
        >>>     return param1 * x + const1 * y

        >>> param_fun = ParamFun(my_function, constants={'const1': 1.0}, parameters_dimensions={'param1': 1})
        >>> result = param_fun(input1, input2)
    """
    @enforce_types
    def __init__(self, param_fun:Callable,
                 constants:list|dict|None = None,
                 parameters_dimensions:list|dict|None = None,
                 parameters:list|dict|None = None,
                 map_over_batch:bool = False) -> Stream:

        self.relation_name = paramfun_relation_name

        # input parameters
        self.param_fun = param_fun
        self.constants = constants
        self.parameters_dimensions = parameters_dimensions
        self.parameters = parameters
        self.map_over_batch = map_over_batch

        self.output_dimension = {}
        super().__init__('F'+paramfun_relation_name + str(NeuObj.count))
        code = textwrap.dedent(inspect.getsource(param_fun)).replace('\"', '\'')
        self.json['Functions'][self.name] = {
            'code' : code,
            'name' : param_fun.__name__,
        }
        self.json['Functions'][self.name]['params_and_consts'] = []

        # Create the missing constants from list
        if type(self.constants) is list:
            for const in self.constants:
                if type(const) is Constant:
                    self.json['Functions'][self.name]['params_and_consts'].append(const.name)
                    #self.json['Constants'][const.name] = copy.deepcopy(const.json['Constants'][const.name])
                    self.json = merge(self.json, const.json)
                elif type(const) is str:
                    self.json['Functions'][self.name]['params_and_consts'].append(const)
                    self.json['Constants'][const] = {'dim': 1, 'sw' : 1}
                    #self.json = merge(self.json, Constant(name=const, values=[[0.0]], sw=1).json)
                else:
                    check(type(const) is Constant or type(const) is str, TypeError,
                          'The element inside the \"constants\" list must be a Constant or str')

        # Create the missing parameters from list
        if type(self.parameters) is list:
            check(self.parameters_dimensions is None, ValueError,'\"parameters_dimensions\" must be None if \"parameters\" is set using list')
            for param in self.parameters:
                if type(param) is Parameter:
                    self.json['Functions'][self.name]['params_and_consts'].append(param.name)
                    self.json = merge(self.json,param.json)
                elif type(param) is str:
                    self.json['Functions'][self.name]['params_and_consts'].append(param)
                    if param not in self.json['Parameters'].keys():
                        self.json = merge(self.json, Parameter(name=param, dimensions=1, sw=1).json)
                else:
                    check(type(param) is Parameter or type(param) is str, TypeError,
                          'The element inside the \"parameters\" list must be a Parameter or str')
        elif type(self.parameters_dimensions) is list:
            funinfo = inspect.getfullargspec(self.param_fun)
            for i, param_dim in enumerate(self.parameters_dimensions):
                idx = i + len(funinfo.args) - len(self.parameters_dimensions)
                param_name = self.name + str(idx)
                self.json['Functions'][self.name]['params_and_consts'].append(param_name)
                self.json = merge(self.json, Parameter(name=param_name, dimensions=list(self.parameters_dimensions[i]), sw=1).json)

        self.json_stream = {}

    @enforce_types
    def __call__(self, *obj:Union[Stream|Parameter|Constant|float|int]) -> Stream:
        stream_name = paramfun_relation_name + str(Stream.count)

        funinfo = inspect.getfullargspec(self.param_fun)
        n_function_input = len(funinfo.args)
        n_call_input = len(obj)
        n_new_constants_and_params = n_function_input - n_call_input

        input_dimensions = []
        input_types = []
        for ind, o in enumerate(obj):
            if type(o) in (int, float, list):
                obj_type = Constant
            else:
                obj_type = type(o)
            o = toStream(o)
            check(type(o) is Stream, TypeError,
                  f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
            input_types.append(obj_type)
            input_dimensions.append(o.dim)

        if n_call_input not in self.json_stream:
            if len(self.json_stream) > 0:
                log.warning(f"The function {self.name} was called with a different number of inputs. If both functions enter in the model an error will be raised.")

            self.json_stream[n_call_input] = copy.deepcopy(self.json)
            self.json_stream[n_call_input]['Functions'][self.name]['n_input'] = n_call_input
            self.__create_missing_parameters(self.json_stream[n_call_input], n_new_constants_and_params)

            # Create the missing parameters
            missing_params = n_new_constants_and_params - len(self.json_stream[n_call_input]['Functions'][self.name]['params_and_consts'])
            check(missing_params == 0, ValueError, f"The function is called with different number of inputs.")

            self.json_stream[n_call_input]['Functions'][self.name]['in_dim'] = copy.deepcopy(input_dimensions)
            self.json_stream[n_call_input]['Functions'][self.name]['map_over_dim'] = self.__infer_map_over_batch(input_types, n_new_constants_and_params)
            output_dimension = self.__infer_output_dimensions(self.json_stream[n_call_input], input_types, input_dimensions)
        else:
            map_over_batch = self.__infer_map_over_batch(input_types, n_new_constants_and_params)
            check(map_over_batch == self.json_stream[n_call_input]['Functions'][self.name]['map_over_dim'], ValueError, f"The function {self.name} was called with different type of input using map_over_batch=True.")
            output_dimension = self.__infer_output_dimensions(self.json_stream[n_call_input], input_types, input_dimensions)

            # Save the all the input dimension used for call the parametric function
            in_dim = self.json_stream[n_call_input]['Functions'][self.name]['in_dim']
            if type(in_dim[0]) is dict:
                if in_dim != input_dimensions:
                    in_dim = [in_dim, input_dimensions]
                    log.warning(f"The function {self.name} was called with inputs with different dimensions.")
            elif input_dimensions not in in_dim:
                in_dim.append(input_dimensions)
                log.warning(f"The function {self.name} was called with inputs with different dimensions.")
            self.json_stream[n_call_input]['Functions'][self.name]['in_dim'] = in_dim

        stream_json = copy.deepcopy(self.json_stream[n_call_input])
        input_names = []
        for ind, o in enumerate(obj):
            o = toStream(o)
            check(type(o) is Stream, TypeError,
                  f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
            stream_json = merge(stream_json, o.json)
            input_names.append(o.name)

        stream_json['Relations'][stream_name] = [paramfun_relation_name, input_names, self.name]
        return Stream(stream_name, stream_json, output_dimension)

    def __infer_map_over_batch(self, input_types, n_constants_and_params):
        input_map_dim = ()

        for elem in input_types:
            if elem in (Parameter, Constant):
                input_map_dim += (None,)
            else:
                input_map_dim += (0,)

        for i in range(n_constants_and_params):
            input_map_dim += (None,)

        if self.map_over_batch:
            return list(input_map_dim)
        else:
            return False

    def __create_missing_parameters(self, stream_json, n_new_constants_and_params):
        funinfo = inspect.getfullargspec(self.param_fun)
        # Create the missing parameters and constants from dict
        missing_params = n_new_constants_and_params - len(stream_json['Functions'][self.name]['params_and_consts'])
        if missing_params or type(self.constants) is dict or type(self.parameters) is dict or type(self.parameters_dimensions) is dict:
            n_input = len(funinfo.args) - missing_params
            n_elem_dict = (len(self.constants if type(self.constants) is dict else [])
                           + len(self.parameters if type(self.parameters) is dict else [])
                           + len(self.parameters_dimensions if type(self.parameters_dimensions) is dict else []))
            for i, key in enumerate(funinfo.args):
                if i >= n_input:
                    if type(self.parameters) is dict and key in self.parameters:
                        if self.parameters_dimensions:
                            check(key in self.parameters_dimensions, TypeError, f'The parameter {key} must be removed from \"parameters_dimensions\".')
                        param = self.parameters[key]
                        if type(self.parameters[key]) is Parameter:
                            stream_json['Functions'][self.name]['params_and_consts'].append(param.name)
                            stream_json['Parameters'][param.name] = copy.deepcopy(param.json['Parameters'][param.name])
                            #stream_json = merge(stream_json, param.json)
                        elif type(self.parameters[key]) is str:
                            stream_json['Functions'][self.name]['params_and_consts'].append(param)
                            stream_json['Parameters'][param] = {'dim' : 1,'sw' : 1}
                            #stream_json = merge(stream_json, Parameter(name=param, dimensions=1, sw=1).json)
                        else:
                            check(type(param) is Parameter or type(param) is str, TypeError,
                                  'The element inside the \"parameters\" dict must be a Parameter or str')
                        n_elem_dict -= 1
                    elif type(self.parameters_dimensions) is dict and key in self.parameters_dimensions:
                        param_name = self.name + key
                        dim = self.parameters_dimensions[key]
                        check(isinstance(dim,(list,tuple,int)), TypeError,
                              'The element inside the \"parameters_dimensions\" dict must be a tuple or int')
                        stream_json['Functions'][self.name]['params_and_consts'].append(param_name)
                        stream_json['Parameters'][param_name] = {'dim': list(dim) if type(dim) is tuple else dim, 'sw' : 1}
                        #json_dim = list(dim) if type(dim) is tuple else dim
                        #stream_json = merge(stream_json, Parameter(name=param_name, dimensions=json_dim, sw=1).json)
                        n_elem_dict -= 1
                    elif type(self.constants) is dict and key in self.constants:
                        const = self.constants[key]
                        if type(self.constants[key]) is Constant:
                            stream_json['Functions'][self.name]['params_and_consts'].append(const.name)
                            stream_json['Constants'][const.name] = copy.deepcopy(const.json['Constants'][const.name])
                        elif type(self.constants[key]) is str:
                            stream_json['Functions'][self.name]['params_and_consts'].append(const)
                            stream_json['Constants'][const] = {'dim': 1, 'sw' : 1}
                        else:
                            check(type(const) is Constant or type(const) is str, TypeError,
                                  'The element inside the \"constants\" dict must be a Constant or str')
                        n_elem_dict -= 1
                    else:
                        param_name = self.name + key
                        stream_json['Functions'][self.name]['params_and_consts'].append(param_name)
                        stream_json['Parameters'][param_name] = {'dim': 1, 'sw' : 1}
                        #stream_json = merge(stream_json, Parameter(name=param_name, dimensions=1, sw=1).json)
            check(n_elem_dict == 0, ValueError, 'Some of the input parameters are not used in the function.')

    def __infer_output_dimensions(self, stream_json, input_types, input_dimensions):
        import torch
        batch_dim = 5

        all_inputs_dim = copy.deepcopy(input_dimensions)
        all_inputs_type = copy.deepcopy(input_types)
        params_and_consts = stream_json['Constants'] | stream_json['Parameters']
        for name in stream_json['Functions'][self.name]['params_and_consts']:
            all_inputs_dim.append(params_and_consts[name])
            all_inputs_type.append(Constant)

        n_samples_sec = 0.1
        is_int = False
        while is_int == False:
            n_samples_sec *= 10
            vect_input_time = [math.isclose(d['tw']*n_samples_sec,round(d['tw']*n_samples_sec)) for d in all_inputs_dim if 'tw' in d]
            if len(vect_input_time) == 0:
                is_int = True
            else:
                is_int = sum(vect_input_time) == len(vect_input_time)

        # Build input with right dimensions
        inputs = []
        inputs_win_type = []
        inputs_win = []

        for t, dim in zip(all_inputs_type,all_inputs_dim):
            window = 'tw' if 'tw' in dim else ('sw' if 'sw' in dim else None)
            if window == 'tw':
                dim_win = round(dim[window] * n_samples_sec)
            elif window == 'sw':
                dim_win = dim[window]
            else:
                dim_win = 1
            if t in (Parameter, Constant):
                if type(dim['dim']) is list:
                    inputs.append(torch.rand(size=(dim_win,) + tuple(dim['dim'])))
                else:
                    inputs.append(torch.rand(size=(dim_win, dim['dim'])))
            else:
                inputs.append(torch.rand(size=(batch_dim, dim_win, dim['dim'])))

            inputs_win_type.append(window)
            inputs_win.append(dim_win)

        if self.map_over_batch:
            function_to_call = torch.func.vmap(self.param_fun,in_dims=tuple(stream_json['Functions'][self.name]['map_over_dim']))
        else:
            function_to_call = self.param_fun
        out = function_to_call(*inputs)
        out_shape = out.shape
        check(out_shape[0] == batch_dim, ValueError, "The batch output dimension it is not correct.")
        out_dim = list(out_shape[2:])
        check(len(out_dim) == 1, ValueError, "The output dimension of the function is bigger than a vector.")
        out_win_type = 'sw'
        out_win = out_shape[1]
        for idx, win in enumerate(inputs_win):
            if out_shape[1] == win and all_inputs_type[idx] not in (Parameter, Constant):
                out_win_type = inputs_win_type[idx]
                out_win = all_inputs_dim[idx][out_win_type]

        return { 'dim': out_dim[0], out_win_type : out_win }

def return_standard_inputs(json, model_def, xlim = None, num_points = 1000):
    check(json['n_input'] == 1 or json['n_input'] == 2, ValueError, "The function must have only one or two inputs.")
    fun_inputs = tuple()
    for i in range(json['n_input']):
        dim = json['in_dim'][i]
        check(dim['dim'] == 1, ValueError, "The input dimension must be 1.")
        if 'tw' in dim:
            check(dim['tw'] == model_def['Info']['SampleTime'], ValueError, "The input window must be 1.")
        elif 'sw' in dim:
            check(dim['sw'] == 1, ValueError, "The input window must be 1.")
        if xlim is not None:
            if json['n_input'] == 2:
                check(np.array(xlim).shape == (json['n_input'], 2), ValueError,
                      "The xlim must have the same shape as the number of inputs.")
                x_value = np.linspace(xlim[i][0], xlim[i][1], num=num_points)
            else:
                check(np.array(xlim).shape == (2,), ValueError,
                      "The xlim must have the same shape as the number of inputs.")
                x_value = np.linspace(xlim[0], xlim[1], num=num_points)
        else:
            x_value = np.linspace(0, 1, num=num_points)
        if i == 0:
            x0_value = torch.from_numpy(x_value)
        else:
            x1_value = torch.from_numpy(x_value)

    if json['n_input'] == 2:
        x0_value, x1_value = torch.meshgrid(x0_value,x1_value,indexing="xy")
        x0_value = x0_value.flatten().unsqueeze(1).unsqueeze(1)
        x1_value = x1_value.flatten().unsqueeze(1).unsqueeze(1)
        fun_inputs += (x0_value,x1_value,)
    else:
        x0_value = x0_value.unsqueeze(1).unsqueeze(1)
        fun_inputs += (x0_value,)

    for key in json['params_and_consts']:
        val = model_def['Parameters'][key] if key in model_def['Parameters'] else model_def['Constants'][key]
        fun_inputs += tuple([torch.from_numpy(np.array(val['values']))]) # The vector is transform in a tuple

    return fun_inputs

def return_function(json, fun_inputs):
    exec(json['code'], globals())
    function_to_call = globals()[json['name']]
    output = function_to_call(*fun_inputs)
    check(output.shape[1] == 1, ValueError, "The output dimension must be 1.")
    check(output.shape[2] == 1, ValueError, "The output window must be 1.")
    funinfo = inspect.getfullargspec(function_to_call)
    return output, funinfo.args

class Parametric_Layer(nn.Module):
    def __init__(self, func, params_and_consts, map_over_batch):
        super().__init__()
        self.name = func['name']
        self.params_and_consts = params_and_consts
        if type(map_over_batch) is list:
            self.map_over_batch = True
            self.input_map_dim = tuple(map_over_batch)
        else:
            self.map_over_batch = False
        ## Add the function to the globals
        try:
            code = 'import torch\n@torch.fx.wrap\n' + func['code']
            exec(code, globals())
        except Exception as e:
            print(f"An error occurred: {e}")

    def forward(self, *inputs):
        args = list(inputs) + self.params_and_consts
        # Retrieve the function object from the globals dictionary
        function_to_call = globals()[self.name]
        # Call the function using the retrieved function object
        if self.map_over_batch:
            function_to_call = torch.func.vmap(function_to_call,in_dims=self.input_map_dim)
        result = function_to_call(*args)
        return result

def createParamFun(self, *func_params):
    return Parametric_Layer(func=func_params[0], params_and_consts=func_params[1], map_over_batch=func_params[2])

setattr(Model, paramfun_relation_name, createParamFun)
