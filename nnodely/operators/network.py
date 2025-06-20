import copy

import numpy as np
import  torch, random

from nnodely.support.utils import TORCH_DTYPE, check, enforce_types, tensor_to_list
from nnodely.basic.modeldef import ModelDef

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

class Network:
    @enforce_types
    def __init__(self):
        check(type(self) is not Network, TypeError, "Loader class cannot be instantiated directly")

        # Models definition
        self._model_def = ModelDef()
        self._model = None
        self._neuralized = False
        self._traced = False

        # Model components
        self._states = {}
        self._input_n_samples = {}
        self._input_ns_backward = {}
        self._input_ns_forward = {}
        self._max_samples_backward = None
        self._max_samples_forward = None
        self._max_n_samples = 0

        # Dataset information
        self._data_loaded = False
        self._file_count = 0
        self._num_of_samples = {}
        self._data = {}
        self._multifile = {}

        # Training information
        self._training = {}

        # Save internal
        self._log_internal = False
        self._internals = {}

    def _save_internal(self, key, value):
        self._internals[key] = tensor_to_list(value)

    def _set_log_internal(self, log_internal:bool):
        self._log_internal = log_internal

    def _clean_log_internal(self):
        self._internals = {}

    def _removeVirtualStates(self, connect, closed_loop):
        if connect or closed_loop:
            for key in (connect.keys() | closed_loop.keys()):
                if key in self._states.keys():
                    del self._states[key]

    def _updateState(self, X, out_closed_loop, out_connect):
        for key, value in out_connect.items():
            X[key] = value
            self._states[key] = X[key].clone().detach()
        for key, val in out_closed_loop.items():
            shift = val.shape[1]  ## take the output time dimension
            X[key] = torch.roll(X[key], shifts=-1, dims=1)  ## Roll the time window
            X[key][:, -shift:, :] = val  ## substitute with the predicted value
            self._states[key] = X[key].clone().detach()

    def _get_gradient_on_inference(self):
        for key, value in self._model_def['Inputs'].items():
            if 'type' in value.keys():
                return True
        return False

    def _get_mandatory_inputs(self, connect, closed_loop):
        model_inputs = list(self._model_def['Inputs'].keys())
        non_mandatory_inputs = \
            list(closed_loop.keys()) + list(connect.keys()) + list(self._model_def.recurrentInputs().keys())
        mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))
        return mandatory_inputs, non_mandatory_inputs

    def _get_not_mandatory_inputs(self, data, X, non_mandatory_inputs, remaning_indexes, batch_size, step, shuffle = False):
        related_indexes = random.sample(remaning_indexes, batch_size) if shuffle else remaning_indexes[:batch_size]
        for num in related_indexes:
            remaning_indexes.remove(num)
        if step > 0:
            if len(remaning_indexes) >= step:
                step_idxs = random.sample(remaning_indexes, step) if shuffle else remaning_indexes[:step]
                for num in step_idxs:
                    remaning_indexes.remove(num)
            else:
                remaning_indexes.clear()

        for key in non_mandatory_inputs:
            if key in data.keys(): ## with data
                X[key] = data[key][related_indexes]
            else:  ## with zeros
                window_size = self._input_n_samples[key]
                dim = self._model_def['Inputs'][key]['dim']
                if 'type' in self._model_def['Inputs'][key]:
                    X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=TORCH_DTYPE, requires_grad=True)
                else:
                    X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)
                self._states[key] = X[key]
        return related_indexes

    def _inference(self, data, n_samples, batch_size, loss_gains, loss_functions,
                    shuffle = False, optimizer = None,
                    total_losses = None, A = None, B = None):
        if shuffle:
            randomize = torch.randperm(n_samples)
            data = {key: val[randomize] for key, val in data.items()}
        ## Initialize the train losses vector
        aux_losses = torch.zeros([len(self._model_def['Minimizers']), n_samples // batch_size])
        for idx in range(0, (n_samples - batch_size + 1), batch_size):
            ## Build the input tensor
            XY = {key: val[idx:idx + batch_size] for key, val in data.items()}
            ## Reset gradient
            if optimizer:
                optimizer.zero_grad()
            ## Model Forward
            _, minimize_out, _, _ = self._model(XY)  ## Forward pass
            ## Loss Calculation
            total_loss = 0
            for ind, (key, value) in enumerate(self._model_def['Minimizers'].items()):
                if A is not None:
                    A[key].append(minimize_out[value['A']].detach().numpy())
                if B is not None:
                    B[key].append(minimize_out[value['B']].detach().numpy())
                loss = loss_functions[key](minimize_out[value['A']], minimize_out[value['B']])
                loss = (loss * loss_gains[key]) if key in loss_gains.keys() else loss
                if total_losses is not None:
                    total_losses[key].append(loss.detach().numpy())
                aux_losses[ind][idx // batch_size] = loss.item()
                total_loss += loss
            ## Gradient step
            if optimizer:
                total_loss.backward()
                optimizer.step()
                self.visualizer.showWeightsInTrain(batch=idx // batch_size)

        ## return the losses
        return aux_losses

    def _recurrent_inference(self, data, batch_indexes, batch_size, loss_gains, prediction_samples,
                             step, non_mandatory_inputs, mandatory_inputs, loss_functions,
                             shuffle = False, optimizer = None,
                             total_losses = None, A = None, B = None):
        indexes = copy.deepcopy(batch_indexes)
        aux_losses = \
            torch.zeros([len(self._model_def['Minimizers']), round((len(indexes) + step) / (batch_size + step))])
        X = {}
        batch_val = 0
        while len(indexes) >= batch_size:
            selected_indexes = self._get_not_mandatory_inputs(data, X, non_mandatory_inputs, indexes, batch_size, step, shuffle)

            horizon_losses = {ind: [] for ind in range(len(self._model_def['Minimizers']))}
            if optimizer:
                optimizer.zero_grad()  ## Reset the gradient

            for horizon_idx in range(prediction_samples + 1):
                ## Get data
                for key in mandatory_inputs:
                    X[key] = data[key][[idx + horizon_idx for idx in selected_indexes]]
                ## Forward pass
                out, minimize_out, out_closed_loop, out_connect = self._model(X)

                if self._log_internal:
                    #assert (check_gradient_operations(self._states) == 0)
                    #assert (check_gradient_operations(data) == 0)
                    internals_dict = {'XY': tensor_to_list(X), 'out': out, 'param': self._model.all_parameters,
                                      'closedLoop': self._model.closed_loop_update, 'connect': self._model.connect_update}

                ## Loss Calculation
                for ind, (key, value) in enumerate(self._model_def['Minimizers'].items()):
                    if A is not None:
                        A[key][horizon_idx].append(minimize_out[value['A']].detach().numpy())
                    if B is not None:
                        B[key][horizon_idx].append(minimize_out[value['B']].detach().numpy())
                    loss = loss_functions[key](minimize_out[value['A']], minimize_out[value['B']])
                    loss = (loss * loss_gains[key]) if key in loss_gains.keys() else loss
                    horizon_losses[ind].append(loss)

                ## Update
                self._updateState(X, out_closed_loop, out_connect)

                if self._log_internal:
                    internals_dict['state'] = self._states
                    self._save_internal('inout_' + str(batch_val) + '_' + str(horizon_idx), internals_dict)

            ## Calculate the total loss
            total_loss = 0
            for ind in range(len(self._model_def['Minimizers'])):
                if self.run_training_params['weights_function'] is not None:
                    # TODO: check if the weights function is correct (types, return type, dimensions, etc.)
                    weights = self.run_training_params['weights_function'](len(horizon_losses[ind]))
                    loss = torch.sum(torch.stack(horizon_losses[ind]) * weights)/torch.sum(weights) # weighted average
                    #loss = torch.sum(torch.stack(horizon_losses[ind]) * weights)
                else:
                    loss = sum(horizon_losses[ind]) / (prediction_samples + 1)
                aux_losses[ind][batch_val] = loss.item()
                total_loss += loss

            ## Gradient Step
            if optimizer:
                total_loss.backward()  ## Backpropagate the error
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                self.visualizer.showWeightsInTrain(batch=batch_val)
            batch_val += 1

        ## return the losses
        return aux_losses

    def _setup_recurrent_variables(self, prediction_samples, closed_loop, connect):
        ## Prediction samples
        check(prediction_samples >= -1, KeyError, 'The sample horizon must be positive or -1 for disconnect connection!')

        ## Close loop information
        for input, output in closed_loop.items():
            check(input in self._model_def['Inputs'], ValueError, f'the tag {input} is not an input variable.')
            check(output in self._model_def['Outputs'], ValueError,
                  f'the tag {output} is not an output of the network')
            log.warning(
                f'Recurrent train: closing the loop between the the input ports {input} and the output ports {output} for {prediction_samples} samples')

        ## Connect information
        for connect_in, connect_out in connect.items():
            check(connect_in in self._model_def['Inputs'], ValueError,
                  f'the tag {connect_in} is not an input variable.')
            check(connect_out in self._model_def['Outputs'], ValueError,
                  f'the tag {connect_out} is not an output of the network')
            log.warning(
                f'Recurrent train: connecting the input ports {connect_in} with output ports {connect_out} for {prediction_samples} samples')

        ## Disable recurrent training if there are no recurrent variables
        if len(connect|closed_loop|self._model_def.recurrentInputs()) == 0:
            if prediction_samples >= 0:
                log.warning(
                    f"The value of the prediction_samples={prediction_samples} but the network has no recurrent variables.")
            prediction_samples = -1

        return prediction_samples

    @enforce_types
    def resetStates(self, states:set={}, batch:int=1) -> None:
        if states: ## reset only specific states
            for key in states:
                window_size = self._input_n_samples[key]
                dim = self._model_def['Inputs'][key]['dim']
                self._states[key] = torch.zeros(size=(batch, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)
        else: ## reset all states
            self._states = {}
            for key, state in self._model_def.recurrentInputs().items():
                window_size = self._input_n_samples[key]
                dim = state['dim']
                self._states[key] = torch.zeros(size=(batch, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)

