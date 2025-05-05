import copy

import numpy as np

from nnodely.support.utils import check, merge, subjson_from_model, subjson_from_output
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
        self._input_connect = {}
        self._input_closed_loop = {}

    def __contains__(self, key):
        return key in self.__json

    def __getitem__(self, key):
        return self.__json[key]

    def __setitem__(self, key, value):
        self.__json[key] = value

    def __checkModel(self, json):
        all_inputs = json['Inputs'].keys()
        all_outputs = json['Outputs'].keys()

        subjson = MAIN_JSON
        for name in all_outputs:
            subjson = merge(subjson, subjson_from_output(json, name))
        needed_inputs = subjson['Inputs'].keys()

        extenal_inputs = set(all_inputs) - set(needed_inputs)
        check(all_inputs == needed_inputs, RuntimeError,
              f'Connect or close loop operation on the inputs {list(extenal_inputs)}, that are not used in the model.')

    def recurrentInputs(self):
        return {key:value for key, value in self.__json['Inputs'].items() if ('closedLoop' in value.keys() or 'connect' in value.keys())}

    def getJson(self, models:list|str|None = None) -> dict:
        if models is None:
            return copy.deepcopy(self.__json)
        else:
            json = subjson_from_model(self.__json, models)
            self.__checkModel(json)
            return copy.deepcopy(json)

    def getSampleTime(self):
        check(self.__sample_time is not None, AttributeError, "Sample time is not defined the model is not neuralized!")
        return self.__sample_time

    def isDefined(self):
        return self.__json is not None

    def update(self, model_def = None, model_dict = None, minimize_dict = None, update_state_dict = None):
        self.__json = copy.deepcopy(model_def) if model_def is not None else copy.deepcopy(self.__json_base)
        model_dict = copy.deepcopy(model_dict) if model_dict is not None else self.__model_dict
        minimize_dict = copy.deepcopy(minimize_dict) if minimize_dict is not None else self.__minimize_dict

        # Add models to the model_def
        for key, stream_list in model_dict.items():
            for stream in stream_list:
                self.__json = merge(self.__json, stream.json)
        if len(model_dict) > 1:
            if 'Models' not in self.__json:
                self.__json['Models'] = {}
            for model_name, model_params in model_dict.items():
                self.__json['Models'][model_name] = {
                    'Inputs': [], 'Outputs': [], 'Parameters': [], 'Constants': [], 'Functions': [], 'Relations': []}
                parameters, constants, inputs, functions, relations = set(), set(), set(), set(), set()
                for param in model_params:
                    self.__json['Models'][model_name]['Outputs'].append(param.name)
                    parameters |= set(param.json['Parameters'].keys())
                    constants |= set(param.json['Constants'].keys())
                    inputs |= set(param.json['Inputs'].keys())
                    functions |= set(param.json['Functions'].keys())
                    relations |= set(param.json['Relations'].keys())
                self.__json['Models'][model_name]['Parameters'] = list(parameters)
                self.__json['Models'][model_name]['Constants'] = list(constants)
                self.__json['Models'][model_name]['Inputs'] = list(inputs)
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

        if "SampleTime" in self.__json['Info']:
            self.__sample_time = self.__json['Info']["SampleTime"]

    def addConnect(self, stream_out, input_list_in):
        if type(input_list_in) is not list:
            input_list_in = [input_list_in]
        for input in input_list_in:
            self._input_connect[input.name] = stream_out.name

    def addClosedLoop(self, stream_out, input_list_in):
        if type(input_list_in) is not list:
            input_list_in = [input_list_in]
        for input in input_list_in:
            self._input_closed_loop[input.name] = stream_out.name

    def addModel(self, name, stream_list):
        if isinstance(stream_list, (Output,Stream)):
            stream_list = [stream_list]

        subjson = MAIN_JSON
        json = MAIN_JSON
        for stream in stream_list:
            subjson = merge(subjson, subjson_from_output(stream.json, stream.name))
            json = merge(json, stream.json)

        all_inputs = json['Inputs'].keys()
        needed_inputs = subjson['Inputs'].keys()

        extenal_inputs = set(all_inputs) - set(needed_inputs)
        check(all_inputs == needed_inputs, RuntimeError, f'Connect or close loop operation on the inputs {list(extenal_inputs)}, that are not used in the model.')

        if type(stream_list) is list:
            check(name not in self.__model_dict.keys(), ValueError, f"The name '{name}' of the model is already used")
            self.__model_dict[name] = copy.deepcopy(stream_list)
        else:
            raise TypeError(f'stream_list is type {type(stream_list)} but must be an Output or Stream or a list of them')

        try:
            self.update()
        except Exception as e:
            self.removeModel(name)
            raise e

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

        check(self.__json['Inputs'] != {}, RuntimeError, "No model is defined!")
        json_inputs = self.__json['Inputs']

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