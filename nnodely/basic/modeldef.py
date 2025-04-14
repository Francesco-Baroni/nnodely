import copy

import numpy as np

from nnodely.support.utils import check, merge
from nnodely.basic.relation import MAIN_JSON, Stream
from nnodely.layers.output import Output

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)

class ModelDef():
    def __init__(self, model_def = MAIN_JSON):
        # Models definition
        self.__json_base = copy.deepcopy(model_def)

        # Inizialize the model definition
        self.__json = copy.deepcopy(self.__json_base)
        if "SampleTime" in self.__json['Info']:
            self.__sample_time = self.__json['Info']["SampleTime"]
        else:
            self.__sample_time = None
        self.__model_dict = {}
        self.__minimize_dict = {}
        self.__update_state_dict = {}

    def __contains__(self, key):
        return key in self.__json

    def __getitem__(self, key):
        return self.__json[key]

    def __setitem__(self, key, value):
        self.__json[key] = value

    #TODO to remove when getJson takes a model list as argment
    def getModelDict(self):
        return copy.deepcopy(self.__model_dict)

    def getJson(self):
        return copy.deepcopy(self.__json)

    def getSampleTime(self):
        check(self.__sample_time is not None, AttributeError, "Sample time is not defined the model is not neuralized!")
        return self.__sample_time

    def isDefined(self):
        return self.__json is not None

    def update(self, model_def = None, model_dict = None, minimize_dict = None, update_state_dict = None):
        self.__json = copy.deepcopy(model_def) if model_def is not None else copy.deepcopy(self.__json_base)
        model_dict = copy.deepcopy(model_dict) if model_dict is not None else self.__model_dict
        minimize_dict = copy.deepcopy(minimize_dict) if minimize_dict is not None else self.__minimize_dict
        update_state_dict = copy.deepcopy(update_state_dict) if update_state_dict is not None else self.__update_state_dict

        # Add models to the model_def
        for key, stream_list in model_dict.items():
            for stream in stream_list:
                self.__json = merge(self.__json, stream.json)
        if len(model_dict) > 1:
            if 'Models' not in self.__json:
                self.__json['Models'] = {}
            for model_name, model_params in model_dict.items():
                self.__json['Models'][model_name] = {
                    'Inputs': [], 'States': [], 'Outputs': [],
                    'Parameters': [],'Constants': [],'Functions': [], 'Relations': []}
                parameters, constants, inputs, states, functions, relations = set(), set(), set(), set(), set(), set()
                for param in model_params:
                    self.__json['Models'][model_name]['Outputs'].append(param.name)
                    parameters |= set(param.json['Parameters'].keys())
                    constants |= set(param.json['Constants'].keys())
                    inputs |= set(param.json['Inputs'].keys())
                    states |= set(param.json['States'].keys())
                    functions |= set(param.json['Functions'].keys())
                    relations |= set(param.json['Relations'].keys())
                self.__json['Models'][model_name]['Parameters'] = list(parameters)
                self.__json['Models'][model_name]['Constants'] = list(constants)
                self.__json['Models'][model_name]['Inputs'] = list(inputs)
                self.__json['Models'][model_name]['States'] = list(states)
                self.__json['Models'][model_name]['Functions'] = list(functions)
                self.__json['Models'][model_name]['Relations'] = list(relations)
        elif len(model_dict) == 1:
            self.__json['Models'] = list(model_dict.keys())[0]

        if 'Minimizers' not in self.__json:
            self.__json['Minimizers'] = {}
        for key, minimize in minimize_dict.items():
            self.__json = merge(self.__json, minimize['A'].json)
            self.__json = merge(self.__json, minimize['B'].json)
            self.__json['Minimizers'][key] = {}
            self.__json['Minimizers'][key]['A'] = minimize['A'].name
            self.__json['Minimizers'][key]['B'] = minimize['B'].name
            self.__json['Minimizers'][key]['loss'] = minimize['loss']

        for key, update_state in update_state_dict.items():
            self.__json = merge(self.__json, update_state.json)

        if "SampleTime" in self.__json['Info']:
            self.__sample_time = self.__json['Info']["SampleTime"]

    def __update_state(self, stream_out, state_list_in, UpdateState):
        from nnodely.layers.input import  State
        if type(state_list_in) is not list:
            state_list_in = [state_list_in]
        for state_in in state_list_in:
            check(isinstance(stream_out, (Output, Stream)), TypeError,
                  f"The {stream_out} must be a Stream or Output and not a {type(stream_out)}.")
            check(type(state_in) is State, TypeError,
                  f"The {state_in} must be a State and not a {type(state_in)}.")
            check(stream_out.dim['dim'] == state_in.dim['dim'], ValueError,
                  f"The dimension of {stream_out.name} is not equal to the dimension of {state_in.name} ({stream_out.dim['dim']}!={state_in.dim['dim']}).")
            if type(stream_out) is Output:
                stream_name = self.__json['Outputs'][stream_out.name]
                stream_out = Stream(stream_name,stream_out.json,stream_out.dim, 0)
            self.__update_state_dict[state_in.name] = UpdateState(stream_out, state_in)

    def addConnect(self, stream_out, state_list_in):
        from nnodely.layers.input import Connect
        self.__update_state(stream_out, state_list_in, Connect)
        self.update()

    def addClosedLoop(self, stream_out, state_list_in):
        from nnodely.layers.input import ClosedLoop
        self.__update_state(stream_out, state_list_in, ClosedLoop)
        self.update()

    def addModel(self, name, stream_list):
        if isinstance(stream_list, (Output,Stream)):
            stream_list = [stream_list]
        if type(stream_list) is list:
            check(name not in self.__model_dict.keys(), ValueError, f"The name '{name}' of the model is already used")
            self.__model_dict[name] = copy.deepcopy(stream_list)
        else:
            raise TypeError(f'stream_list is type {type(stream_list)} but must be an Output or Stream or a list of them')
        self.update()

    def removeModel(self, name_list):
        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in self.__model_dict, IndexError, f"The name {name} is not part of the available models")
                del self.__model_dict[name]
        self.update()

    def addMinimize(self, name, streamA, streamB, loss_function='mse'):
        check(isinstance(streamA, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
        check(isinstance(streamB, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
        #check(streamA.dim == streamB.dim, ValueError, f'Dimension of streamA={streamA.dim} and streamB={streamB.dim} are not equal.')
        self.__minimize_dict[name]={'A':copy.deepcopy(streamA), 'B': copy.deepcopy(streamB), 'loss':loss_function}
        self.update()

    def removeMinimize(self, name_list):
        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in self.__minimize_dict, IndexError, f"The name {name} is not part of the available minimuzes")
                del self.__minimize_dict[name]
        self.update()

    def setBuildWindow(self, sample_time = None):
        check(self.__json is not None, RuntimeError, "No model is defined!")
        if sample_time is not None:
            check(sample_time > 0, RuntimeError, 'Sample time must be strictly positive!')
            self.__sample_time = sample_time
        else:
            if self.__sample_time is None:
                self.__sample_time = 1

        self.__json['Info'] = {"SampleTime": self.__sample_time}

        check(self.__json['Inputs'] | self.__json['States'] != {}, RuntimeError, "No model is defined!")
        json_inputs = self.__json['Inputs'] | self.__json['States']

        # for key,value in self.json['States'].items():
        #     check(closedloop_name in self.json['States'][key].keys() or connect_name in self.json['States'][key].keys(),
        #           KeyError, f'Update function is missing for state {key}. Use Connect or ClosedLoop to update the state.')

        input_tw_backward, input_tw_forward, input_ns_backward, input_ns_forward = {}, {}, {}, {}
        for key, value in json_inputs.items():
            if value['sw'] == [0,0] and value['tw'] == [0,0]:
                assert(False), f"Input '{key}' has no time window or sample window"
            if value['sw'] == [0, 0] and self.__sample_time is not None:
                ## check if value['tw'] is a multiple of sample_time
                absolute_tw = abs(value['tw'][0]) + abs(value['tw'][1])
                check(round(absolute_tw % self.__sample_time) == 0, ValueError,
                      f"Time window of input '{key}' is not a multiple of sample time. This network cannot be neuralized")
                input_ns_backward[key] = round(-value['tw'][0] / self.__sample_time)
                input_ns_forward[key] = round(value['tw'][1] / self.__sample_time)
            elif self.__sample_time is not None:
                input_ns_backward[key] = max(round(-value['tw'][0] / self.__sample_time), -value['sw'][0])
                input_ns_forward[key] = max(round(value['tw'][1] / self.__sample_time), value['sw'][1])
            else:
                check(value['tw'] == [0,0], RuntimeError, f"Sample time is not defined for input '{key}'")
                input_ns_backward[key] = -value['sw'][0]
                input_ns_forward[key] = value['sw'][1]
            value['ns'] = [input_ns_backward[key], input_ns_forward[key]]
            value['ntot'] = sum(value['ns'])

        self.__json['Info']['ns'] = [max(input_ns_backward.values()), max(input_ns_forward.values())]
        self.__json['Info']['ntot'] = sum(self.__json['Info']['ns'])
        if self.__json['Info']['ns'][0] < 0:
            log.warning(
                f"The input is only in the far past the max_samples_backward is: {self.__json['Info']['ns'][0]}")
        if self.__json['Info']['ns'][1] < 0:
            log.warning(
                f"The input is only in the far future the max_sample_forward is: {self.__json['Info']['ns'][1]}")

        for k, v in (self.__json['Parameters'] | self.__json['Constants']).items():
            if 'values' in v:
                window = 'tw' if 'tw' in v.keys() else ('sw' if 'sw' in v.keys() else None)
                if window == 'tw':
                    check(np.array(v['values']).shape[0] == v['tw'] / self.__sample_time, ValueError,
                      f"{k} has a different number of values for this sample time.")
                if v['values'] == "SampleTime":
                    v['values'] = self.__sample_time

    def updateParameters(self, model):
        if model is not None:
            for key in self.__json['Parameters'].keys():
                if key in model.all_parameters:
                    self.__json['Parameters'][key]['values'] = model.all_parameters[key].tolist()
                    if 'init_fun' in self.__json['Parameters'][key]:
                        del self.__json['Parameters'][key]['init_fun']