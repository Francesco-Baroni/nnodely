import copy

import numpy as np

from nnodely.support.utils import check, merge, subjson_from_model, subjson_from_output, subjson_from_relation
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
        # self.__model_dict = {}
        # self.__minimize_dict = {}
        #self._input_connect = {}
        #self._input_closed_loop = {}

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

    # def update(self, model_def = None, model_dict = None, minimize_dict = None, update_state_dict = None):
    #     self.__json = copy.deepcopy(model_def) if model_def is not None else copy.deepcopy(self.__json_base)
    #     model_dict = copy.deepcopy(model_dict) if model_dict is not None else self.__model_dict
    #     minimize_dict = copy.deepcopy(minimize_dict) if minimize_dict is not None else self.__minimize_dict
    #
    #     # Add models to the model_def
    #     for key, stream_list in model_dict.items():
    #         for stream in stream_list:
    #             self.__json = merge(self.__json, stream.json)
    #     if len(model_dict) > 1:
    #         if 'Models' not in self.__json:
    #             self.__json['Models'] = {}
    #         for model_name, model_params in model_dict.items():
    #             self.__json['Models'][model_name] = {
    #                 'Inputs': [], 'Outputs': [], 'Parameters': [], 'Constants': [], 'Functions': [], 'Relations': []}
    #             parameters, constants, inputs, functions, relations = set(), set(), set(), set(), set()
    #             for param in model_params:
    #                 self.__json['Models'][model_name]['Outputs'].append(param.name)
    #                 parameters |= set(param.json['Parameters'].keys())
    #                 constants |= set(param.json['Constants'].keys())
    #                 inputs |= set(param.json['Inputs'].keys())
    #                 functions |= set(param.json['Functions'].keys())
    #                 relations |= set(param.json['Relations'].keys())
    #             self.__json['Models'][model_name]['Parameters'] = list(parameters)
    #             self.__json['Models'][model_name]['Constants'] = list(constants)
    #             self.__json['Models'][model_name]['Inputs'] = list(inputs)
    #             self.__json['Models'][model_name]['Functions'] = list(functions)
    #             self.__json['Models'][model_name]['Relations'] = list(relations)
    #     elif len(model_dict) == 1:
    #         self.__json['Models'] = list(model_dict.keys())[0]
    #
    #     if 'Minimizers' not in self.__json:
    #         self.__json['Minimizers'] = {}
    #     for key, minimize in minimize_dict.items():
    #         self.__json = merge(self.__json, minimize['A'].json)
    #         self.__json = merge(self.__json, minimize['B'].json)
    #         self.__json['Minimizers'][key] = {}
    #         self.__json['Minimizers'][key]['A'] = minimize['A'].name
    #         self.__json['Minimizers'][key]['B'] = minimize['B'].name
    #         self.__json['Minimizers'][key]['loss'] = minimize['loss']
    #
    #     if "SampleTime" in self.__json['Info']:
    #         self.__sample_time = self.__json['Info']["SampleTime"]

    def addConnect(self, stream_out, input_list_in):
        if type(input_list_in) is not list:
            input_list_in = [input_list_in]
        for input in input_list_in:
            # self._input_connect[input.name] = stream_out.name
            outputs = self.__json['Outputs']
            stream_name =  outputs[stream_out.name] if stream_out.name in outputs.keys() else stream_out.name
            self.__json['Inputs'][input.name]['connect'] = stream_name
            #TODO checkif stream_out it is prensent in the __json

    def addClosedLoop(self, stream_out, input_list_in):
        if type(input_list_in) is not list:
            input_list_in = [input_list_in]
        for input in input_list_in:
            # self._input_closed_loop[input.name] = stream_out.name
            outputs = self.__json['Outputs']
            stream_name =  outputs[stream_out.name] if stream_out.name in outputs.keys() else stream_out.name
            self.__json['Inputs'][input.name]['closedLoop'] = stream_name
            # TODO checkif stream_out it is prensent in the __json

    def __get_models_json(self, json):
        model_json = {}
        model_json['Parameters'] = list(json['Parameters'].keys())
        model_json['Constants'] = list(json['Constants'].keys())
        model_json['Inputs'] = list(json['Inputs'].keys())
        model_json['Outputs'] = list(json['Outputs'].keys())
        model_json['Functions'] = list(json['Functions'].keys())
        model_json['Relations'] = list(json['Relations'].keys())
        return model_json

    def addModel(self, name, stream_list):
        if isinstance(stream_list, (Output,Stream)):
            stream_list = [stream_list]

        json = MAIN_JSON
        for stream in stream_list:
            json = merge(json, stream.json)
        self.__checkModel(json)  # TODO Change to warning if the input is outside the model

        if 'Models' not in self.__json:
            self.__json = merge(self.__json, json)
            self.__json['Models'] = name
        else:
            models_names = set(self.__json['Models']) if type(self.__json['Models']) is str else set(self.__json['Models'].keys())
            check(name not in models_names, ValueError,
                  f"The name '{name}' of the model is already used")
            if type(self.__json['Models']) is str:
                self.__json['Models'] = {self.__json['Models']: self.__get_models_json(self.__json)}
            self.__json = merge(self.__json, json)
            self.__json['Models'][name] = self.__get_models_json(json)

        # for stream in stream_list:
        #     self.__json = merge(self.__json, stream.json)
        # self.__checkModel(self.__json)
        #
        # if type(stream_list) is list:
        #     check(name not in self.__model_dict.keys(), ValueError, f"The name '{name}' of the model is already used")
        #     self.__model_dict[name] = copy.deepcopy(stream_list)
        # else:
        #     raise TypeError(f'stream_list is type {type(stream_list)} but must be an Output or Stream or a list of them')

        # try:
        #     self.update()
        # except Exception as e:
        #     self.removeModel(name)
        #     raise e

    def removeModel(self, name_list):
        if 'Models' not in self.__json:
            raise ValueError("No Models are defined")
        models_names = set([self.__json['Models']]) if type(self.__json['Models']) is str else set(self.__json['Models'].keys())

        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in models_names, IndexError, f"The name {name} is not part of the available models")

        models_names -= set(name_list)
        new_json = subjson_from_model(self.__json, list(models_names))

        if 'Minimizers' in self.__json:
            minimizers = set(self.__json['Minimizers'].keys())
            rel_A = [self.__json['Minimizers'][key]['A'] for key in minimizers]
            rel_B = [self.__json['Minimizers'][key]['B'] for key in minimizers]
            relations_name = set(rel_A) | set(rel_B)
            minimizers_json = subjson_from_relation(self.__json, list(relations_name))
            new_json = merge(new_json, minimizers_json)

        self.__json = copy.deepcopy(new_json)

        # if type(name_list) is list:
        #     for name in name_list:
        #         check(name in self.__model_dict, IndexError, f"The name {name} is not part of the available models")
        #         del self.__model_dict[name]
        # self.update()

    def addMinimize(self, name, streamA, streamB, loss_function='mse'):
        check(isinstance(streamA, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
        check(isinstance(streamB, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
        # check(streamA.dim == streamB.dim, ValueError, f'Dimension of streamA={streamA.dim} and streamB={streamB.dim} are not equal.')

        if 'Minimizers' not in self.__json:
            self.__json['Minimizers'] = {}

        streams = merge(streamA.json, streamB.json)
        streamA_name = streamA.json['Outputs'][streamA.name] if isinstance(streamA, Output) else streamA.name
        streamB_name = streamB.json['Outputs'][streamB.name] if isinstance(streamB, Output) else streamB.name
        self.__json = merge(self.__json, streams)
        self.__json['Minimizers'][name] = {}
        self.__json['Minimizers'][name]['A'] = streamA_name
        self.__json['Minimizers'][name]['B'] = streamB_name
        self.__json['Minimizers'][name]['loss'] = loss_function

        # self.__minimize_dict[name]={'A':copy.deepcopy(streamA), 'B': copy.deepcopy(streamB), 'loss':loss_function}
        # self.update()

    def removeMinimize(self, name_list):
        # # TODO to remove a minimizer we need to find the subjson of all the other minimizers and models
        # # Test removeMinimizer
        # if type(name_list) is str:
        #     name_list = [name_list]
        # if type(name_list) is list:
        #     for name in name_list:
        #         check(name in self.__minimize_dict, IndexError, f"The name {name} is not part of the available minimizers")
        #         del self.__minimize_dict[name]
        # self.update()

        if 'Minimizers' not in self.__json:
            raise ValueError("No Minimizers are defined")

        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in self.__json['Minimizers'].keys(), IndexError,
                      f"The name {name} is not part of the available minimizers")

        models_names = set([self.__json['Models']]) if type(self.__json['Models']) is str else set(self.__json['Models'].keys())
        new_json = subjson_from_model(self.__json, list(models_names))

        if 'Minimizers' in self.__json:
            remaning_minimizers = set(self.__json['Minimizers'].keys()) - set(name_list)
            rel_A = [self.__json['Minimizers'][key]['A'] for key in remaning_minimizers]
            rel_B = [self.__json['Minimizers'][key]['B'] for key in remaning_minimizers]
            relations_name = set(rel_A) | set(rel_B)
            if len(relations_name) != 0:
                minimizers_json = subjson_from_relation(self.__json, list(relations_name))
                new_json = merge(new_json, minimizers_json)

        self.__json = copy.deepcopy(new_json)

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

    def updateParameters(self, model = None, *, clear_model = False):
        if clear_model:
            for key in self.__json['Parameters'].keys():
                if 'init_values' in self.__json['Parameters'][key]:
                    self.__json['Parameters'][key]['values'] = self.__json['Parameters'][key]['init_values']
                elif 'values' in self.__json['Parameters'][key]:
                    del self.__json['Parameters'][key]['values']
        elif model is not None:
            for key in self.__json['Parameters'].keys():
                if key in model.all_parameters:
                    self.__json['Parameters'][key]['values'] = model.all_parameters[key].tolist()

    #231