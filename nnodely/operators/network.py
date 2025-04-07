import copy, torch

import numpy as np

from nnodely.basic.modeldef import ModelDef
from nnodely.basic.model import Model
from nnodely.support.utils import check, log, TORCH_DTYPE, NP_DTYPE, argmax_dict, argmin_dict
from nnodely.operators.memory import Memory

class Network(Memory):
    def __init__(self):
        check(type(self) is not Network, TypeError, "Network class cannot be instantiated directly")

    def __addInfo(self):
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model_def['Info']['num_parameters'] = total_params
        from nnodely import __version__
        self.model_def['Info']['nnodely_version'] = __version__

    def addModel(self, name, stream_list):
        """
        Adds a new model with the given name along with a list of Outputs.

        Parameters
        ----------
        name : str
            The name of the model.
        stream_list : list of Stream
            The list of Outputs stream in the model.

        Example
        -------
        Example usage:
            >>> model = Modely()
            >>> x = Input('x')
            >>> out = Output('out', Fir(x.last()))
            >>> model.addModel('example_model', [out])
        """
        try:
            self.model_def.addModel(name, stream_list)
        except Exception as e:
            self.model_def.removeModel(name)
            raise e

    def removeModel(self, name_list):
        """
        Removes models with the given list of names.

        Parameters
        ----------
        name_list : list of str
            The list of model names to remove.

        Example
        -------
        Example usage:
            >>> model.removeModel(['sub_model1', 'sub_model2'])
        """
        self.model_def.removeModel(name_list)

    def addConnect(self, stream_out, state_list_in):
        """
        Adds a connection from a relation stream to an input state.

        Parameters
        ----------
        stream_out : Stream
            The relation stream to connect from.
        state_list_in : list of State
            The list of input states to connect to.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/states.ipynb
            :alt: Open in Colab

        Example:
            >>> model = Modely()
            >>> x = Input('x')
            >>> y = State('y')
            >>> relation = Fir(x.last())
            >>> model.addConnect(relation, y)
        """
        self.model_def.addConnect(stream_out, state_list_in)

    def addClosedLoop(self, stream_out, state_list_in):
        """
        Adds a closed loop connection from a relation stream to an input state.

        Parameters
        ----------
        stream_out : Stream
            The relation stream to connect from.
        state_list_in : list of State
            The list of input states to connect to.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/states.ipynb
            :alt: Open in Colab

        Example:
            >>> model = Modely()
            >>> x = Input('x')
            >>> y = State('y')
            >>> relation = Fir(x.last())
            >>> model.addClosedLoop(relation, y)
        """
        self.model_def.addClosedLoop(stream_out, state_list_in)

    def neuralizeModel(self, sample_time = None, clear_model = False, model_def = None):
        """
        Neuralizes the model, preparing it for inference and training. This method creates a neural network model starting from the model definition.
        It will also create all the time windows for the inputs and states.

        Parameters
        ----------
        sample_time : float or None, optional
            The sample time for the model. Default is None.
        clear_model : bool, optional
            Whether to clear the existing model definition. Default is False.
        model_def : dict or None, optional
            A dictionary defining the model. If provided, it overrides the existing model definition. Default is None.

        Raises
        ------
        ValueError
            If sample_time is not None and model_def is provided.
            If clear_model is True and model_def is provided.

        Example
        -------
        Example usage:
            >>> model = Modely(name='example_model')
            >>> model.neuralizeModel(sample_time=0.1, clear_model=True)
        """
        if model_def is not None:
            check(sample_time == None, ValueError, 'The sample_time must be None if a model_def is provided')
            check(clear_model == False, ValueError, 'The clear_model must be False if a model_def is provided')
            self.model_def = ModelDef(model_def)
        else:
            if clear_model:
                self.model_def.update()
            else:
                self.model_def.updateParameters(self.model)

        for key, state in self.model_def['States'].items():
            check("connect" in state.keys() or  'closedLoop' in state.keys(), KeyError, f'The connect or closed loop missing for state "{key}"')

        self.model_def.setBuildWindow(sample_time)
        self.model = Model(self.model_def.getJson())
        self.__addInfo()

        self._input_ns_backward = {key:value['ns'][0] for key, value in (self.model_def['Inputs']|self.model_def['States']).items()}
        self._input_ns_forward = {key:value['ns'][1] for key, value in (self.model_def['Inputs']|self.model_def['States']).items()}
        self._max_samples_backward = max(self._input_ns_backward.values())
        self._max_samples_forward = max(self._input_ns_forward.values())
        self._input_n_samples = {}
        for key, value in (self.model_def['Inputs'] | self.model_def['States']).items():
            self._input_n_samples[key] = self._input_ns_backward[key] + self._input_ns_forward[key]
        self._max_n_samples = max(self._input_ns_backward.values()) + max(self._input_ns_forward.values())

        ## Initialize States
        self.resetStates()

        self.neuralized = True
        self.traced = False
        self.visualizer.showModel(self.model_def.getJson())
        self.visualizer.showModelInputWindow()
        self.visualizer.showBuiltModel()

    def __call__(self, inputs={}, sampled=False, closed_loop={}, connect={}, prediction_samples='auto',
                 num_of_samples=None):  ##, align_input=False):
        """
        Performs inference on the model.

        Parameters
        ----------
        inputs : dict, optional
            A dictionary of input data. The keys are input names and the values are the corresponding data. Default is an empty dictionary.
        sampled : bool, optional
            A boolean indicating whether the inputs are already sampled. Default is False.
        closed_loop : dict, optional
            A dictionary specifying closed loop connections. The keys are input names and the values are output names. Default is an empty dictionary.
        connect : dict, optional
            A dictionary specifying connections. The keys are input names and the values are output names. Default is an empty dictionary.
        prediction_samples : str or int, optional
            The number of prediction samples. Can be 'auto', None or an integer. Default is 'auto'.
        num_of_samples : str or int, optional
            The number of samples. Can be 'auto', None or an integer. Default is 'auto'.

        Returns
        -------
        dict
            A dictionary containing the model's prediction outputs.

        Raises
        ------
        RuntimeError
            If the network is not neuralized.
        ValueError
            If an input variable is not in the model definition or if an output variable is not in the model definition.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/inference.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> x = Input('x')
            >>> out = Output('out', Fir(x.last()))
            >>> model.addModel('example_model', [out])
            >>> model.neuralizeModel()
            >>> predictions = model(inputs={'x': [1, 2, 3]})
        """

        ## Copy dict for avoid python bug
        inputs = copy.deepcopy(inputs)
        closed_loop = copy.deepcopy(closed_loop)
        connect = copy.deepcopy(connect)

        ## Check neuralize
        check(self.neuralized, RuntimeError, "The network is not neuralized.")

        ## Check closed loop integrity
        for close_in, close_out in (closed_loop | connect).items():
            check(close_in in self.model_def['Inputs'], ValueError, f'the tag "{close_in}" is not an input variable.')
            check(close_out in self.model_def['Outputs'], ValueError,
                  f'the tag "{close_out}" is not an output of the network')

        ## List of keys
        model_inputs = list(self.model_def['Inputs'].keys())
        model_states = list(self.model_def['States'].keys())
        json_inputs = self.model_def['Inputs'] | self.model_def['States']
        state_closed_loop = [key for key, value in self.model_def['States'].items() if
                             'closedLoop' in value.keys()] + list(closed_loop.keys())
        state_connect = [key for key, value in self.model_def['States'].items() if 'connect' in value.keys()] + list(
            connect.keys())
        extra_inputs = list(set(list(inputs.keys())) - set(model_inputs) - set(model_states))
        non_mandatory_inputs = state_closed_loop + state_connect
        mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))

        ## Remove extra inputs
        for key in extra_inputs:
            log.warning(
                f'The provided input {key} is not used inside the network. the inference will continue without using it')
            del inputs[key]

        ## Get the number of data windows for each input/state
        num_of_windows = {key: len(value) for key, value in inputs.items()} if sampled else {
            key: len(value) - self._input_n_samples[key] + 1 for key, value in inputs.items()}

        ## Get the maximum inference window
        if num_of_samples:
            window_dim = num_of_samples
            for key in inputs.keys():
                input_dim = self.model_def['Inputs'][key]['dim'] if key in model_inputs else \
                self.model_def['States'][key]['dim']
                new_samples = num_of_samples - (len(inputs[key]) - self._input_n_samples[key] + 1)
                if input_dim > 1:
                    log.warning(f'The variable {key} is filled with {new_samples} samples equal to zeros.')
                    inputs[key] += [[0 for _ in range(input_dim)] for _ in range(new_samples)]
                else:
                    log.warning(f'The variable {key} is filled with {new_samples} samples equal to zeros.')
                    inputs[key] += [0 for _ in range(new_samples)]
        elif inputs:
            windows = []
            for key in inputs.keys():
                if key in mandatory_inputs:
                    n_samples = len(inputs[key]) if sampled else len(inputs[key]) - self.model_def['Inputs'][key][
                        'ntot'] + 1
                    windows.append(n_samples)
            if not windows:
                for key in inputs.keys():
                    if key in non_mandatory_inputs:
                        if key in model_inputs:
                            n_samples = len(inputs[key]) if sampled else len(inputs[key]) - \
                                                                         self.model_def['Inputs'][key]['ntot'] + 1
                        else:
                            n_samples = len(inputs[key]) if sampled else len(inputs[key]) - \
                                                                         self.model_def['States'][key]['ntot'] + 1
                        windows.append(n_samples)
            window_dim = min(windows) if windows else 0
        else:  ## No inputs
            window_dim = 1 if non_mandatory_inputs else 0
        check(window_dim > 0, StopIteration, f'Missing samples in the input window')

        if len(set(num_of_windows.values())) > 1:
            max_ind_key, max_dim = argmax_dict(num_of_windows)
            min_ind_key, min_dim = argmin_dict(num_of_windows)
            log.warning(
                f'Different number of samples between inputs [MAX {num_of_windows[max_ind_key]} = {max_dim}; MIN {num_of_windows[min_ind_key]} = {min_dim}]')

        ## Autofill the missing inputs
        provided_inputs = list(inputs.keys())
        missing_inputs = list(set(mandatory_inputs) - set(provided_inputs))
        if missing_inputs:
            log.warning(f'Inputs not provided: {missing_inputs}. Autofilling with zeros..')
            for key in missing_inputs:
                inputs[key] = np.zeros(
                    shape=(self._input_n_samples[key] + window_dim - 1, self.model_def['Inputs'][key]['dim']),
                    dtype=NP_DTYPE).tolist()

        ## Transform inputs in 3D Tensors
        for key in inputs.keys():
            input_dim = json_inputs[key]['dim']
            inputs[key] = torch.from_numpy(np.array(inputs[key])).to(TORCH_DTYPE)

            if input_dim > 1:
                correct_dim = 3 if sampled else 2
                check(len(inputs[key].shape) == correct_dim, ValueError,
                      f'The input {key} must have {correct_dim} dimensions')
                check(inputs[key].shape[correct_dim - 1] == input_dim, ValueError,
                      f'The second dimension of the input "{key}" must be equal to {input_dim}')

            if input_dim == 1 and inputs[key].shape[-1] != 1:  ## add the input dimension
                inputs[key] = inputs[key].unsqueeze(-1)
            if inputs[key].ndim <= 1:  ## add the batch dimension
                inputs[key] = inputs[key].unsqueeze(0)
            if inputs[key].ndim <= 2:  ## add the time dimension
                inputs[key] = inputs[key].unsqueeze(0)

        ## initialize the resulting dictionary
        result_dict = {}
        for key in self.model_def['Outputs'].keys():
            result_dict[key] = []

        ## Inference
        calculate_grad = False
        for key, value in json_inputs.items():
            if 'type' in value.keys():
                calculate_grad = True
                break
        with torch.enable_grad() if calculate_grad else torch.inference_mode():
            ## Update with virtual states
            if prediction_samples is not None:
                self.model.update(closed_loop=closed_loop, connect=connect)
            else:
                prediction_samples = 0
            X = {}
            count = 0
            first = True
            for idx in range(window_dim):
                ## Get mandatory data inputs
                for key in mandatory_inputs:
                    X[key] = inputs[key][idx:idx + 1] if sampled else inputs[key][:,
                                                                      idx:idx + self._input_n_samples[key]]
                    if 'type' in json_inputs[key].keys():
                        X[key] = X[key].requires_grad_(True)
                ## reset states
                if count == 0 or prediction_samples == 'auto':
                    count = prediction_samples
                    for key in non_mandatory_inputs:  ## Get non mandatory data (from inputs, from states, or with zeros)
                        ## if prediction_samples is 'auto' and i have enough samples
                        ## if prediction_samples is NOT 'auto' but i have enough extended window (with zeros)
                        if (key in inputs.keys() and prediction_samples == 'auto' and idx < num_of_windows[key]) or (
                                key in inputs.keys() and prediction_samples != 'auto' and idx < inputs[key].shape[1]):
                            X[key] = inputs[key][idx:idx + 1] if sampled else inputs[key][:,
                                                                              idx:idx + self._input_n_samples[key]]
                        ## if im in the first reset
                        ## if i have a state in memory
                        ## if i have prediction_samples = 'auto' and not enough samples
                        elif (key in self._states.keys() and (first or prediction_samples == 'auto')) and (
                                prediction_samples == 'auto' or prediction_samples == None):
                            X[key] = self._states[key]
                        else:  ## if i have no samples and no states
                            window_size = self._input_n_samples[key]
                            dim = json_inputs[key]['dim']
                            X[key] = torch.zeros(size=(1, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)
                            self._states[key] = X[key]
                        if 'type' in json_inputs[key].keys():
                            X[key] = X[key].requires_grad_(True)
                    first = False
                else:
                    # Remove the gradient of the previous forward
                    for key in X.keys():
                        if 'type' in json_inputs[key].keys():
                            X[key] = X[key].detach().requires_grad_(True)
                    count -= 1
                ## Forward pass
                result, _, out_closed_loop, out_connect = self.model(X)

                ## Append the prediction of the current sample to the result dictionary
                for key in self.model_def['Outputs'].keys():
                    if result[key].shape[-1] == 1:
                        result[key] = result[key].squeeze(-1)
                        if result[key].shape[-1] == 1:
                            result[key] = result[key].squeeze(-1)
                    result_dict[key].append(result[key].detach().squeeze(dim=0).tolist())

                ## Update closed_loop and connect
                if prediction_samples:
                    self._updateState(X, out_closed_loop, out_connect)

        ## Remove virtual states
        self._removeVirtualStates(connect, closed_loop)

        return result_dict


