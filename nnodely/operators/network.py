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
        non_mandatory_inputs = list(closed_loop.keys()) + list(connect.keys()) + list(self._model_def.recurrentInputs().keys())
        mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))
        return mandatory_inputs, non_mandatory_inputs
    
    def _get_batch_indexes(self, dataset_name, prediction_samples):
        n_samples = self._num_of_samples[dataset_name]
        #available_samples = n_samples - prediction_samples
        batch_indexes = list(range(n_samples))
        if dataset_name in self._multifile.keys(): ## i have some forbidden indexes
            forbidden_idxs = []
            for i in self._multifile[dataset_name]:
                forbidden_idxs.extend(range(i - prediction_samples, i, 1))
            batch_indexes = [idx for idx in batch_indexes if idx not in forbidden_idxs]
        return batch_indexes
    
    def __split_dataset(self, dataset:str, splits:list, prediction_samples:int):
        check(len(splits) == 3, ValueError, '3 elements must be inserted for the dataset split in training, validation and test')
        check(sum(splits) == 100, ValueError, 'Training, Validation and Test splits must sum up to 100.')
        check(splits[0] > 0, ValueError, 'The training split cannot be zero.')
        check(dataset in self._data.keys(), KeyError, f'{dataset} Not Loaded!') 
        train_size, val_size, test_size = splits[0] / 100, splits[1] / 100, splits[2] / 100
        #dataset = list(dataset) if type(dataset) is str else dataset
        #check(len([data for data in dataset if data in self._data.keys()]) > 0, KeyError, f'the datasets: {dataset} are not loaded!')
        # for data in dataset:
        #     if data not in self._data.keys():
        #         log.warning(f'{data} is not loaded. The training will continue without this dataset.') 
        #         continue
        num_of_samples = self._num_of_samples[dataset]
        n_samples_train, n_samples_val, n_samples_test = round(num_of_samples * train_size), round(num_of_samples * val_size), round(num_of_samples * test_size)
        batch_indexes = self._get_batch_indexes(dataset, prediction_samples)
        XY_train, XY_val, XY_test = {}, {}, {}
        train_indexes, val_indexes, test_indexes = [], [], []
        for key, samples in self._data[dataset].items():
            if val_size == 0.0 and test_size == 0.0:  ## we have only training set
                XY_train[key] = torch.from_numpy(samples).to(TORCH_DTYPE)
                train_indexes = batch_indexes
            elif val_size == 0.0 and test_size != 0.0:  ## we have only training and test set
                XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                XY_test[key] = torch.from_numpy(samples[n_samples_train:]).to(TORCH_DTYPE)
                train_indexes = [i for i in batch_indexes if i < n_samples_train]
                test_indexes = [i for i in batch_indexes if i >= n_samples_train]
            elif val_size != 0.0 and test_size == 0.0:  ## we have only training and validation set
                XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                XY_val[key] = torch.from_numpy(samples[n_samples_train:]).to(TORCH_DTYPE)
                train_indexes = [i for i in batch_indexes if i < n_samples_train]
                val_indexes = [i for i in batch_indexes if i >= n_samples_train]
            else:  ## we have training, validation and test set
                XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                XY_val[key] = torch.from_numpy(samples[n_samples_train:-n_samples_test]).to(TORCH_DTYPE)
                XY_test[key] = torch.from_numpy(samples[n_samples_train + n_samples_val:]).to(TORCH_DTYPE)
                train_indexes = [i for i in batch_indexes if i < n_samples_train]
                val_indexes = [i for i in batch_indexes if n_samples_train <= i < n_samples_train + n_samples_val]
                test_indexes = [i for i in batch_indexes if i >= n_samples_train + n_samples_val]
        check(n_samples_train > 0, ValueError, f'The number of train samples {n_samples_train} must be greater than 0.')
        val_indexes = [i-n_samples_train for i in val_indexes]
        test_indexes = [i-n_samples_train-n_samples_val for i in test_indexes]
        if prediction_samples > 0:
            train_indexes = train_indexes[:-prediction_samples]
            val_indexes = val_indexes[:-prediction_samples]
            test_indexes = test_indexes[:-prediction_samples]
        return XY_train, XY_val, XY_test, n_samples_train, n_samples_val, n_samples_test, train_indexes, val_indexes, test_indexes

    def __get_data(self, train_dataset, validation_dataset=None, test_dataset=None, prediction_samples=0):
        ## Get the names of the datasets
        datasets = list(self._data.keys())
        n_samples_train, n_samples_val, n_samples_test = 0, 0, 0
        check(train_dataset in datasets, KeyError, f'{train_dataset} Not Loaded!')
        if validation_dataset and validation_dataset not in datasets:
            log.warning(f'Validation Dataset [{validation_dataset}] Not Loaded.')
        if test_dataset and test_dataset not in datasets:
            log.warning(f'Test Dataset [{test_dataset}] Not Loaded.')

        n_samples_train, n_samples_val, n_samples_test = 0, 0, 0
        XY_train, XY_val, XY_test = {}, {}, {}
        train_indexes, val_indexes, test_indexes = [], [], []
        ## Split into train, validation and test
        n_samples_train = self._num_of_samples[train_dataset]
        XY_train = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in self._data[train_dataset].items()}
        train_indexes = self._get_batch_indexes(train_dataset, prediction_samples)
        if validation_dataset in datasets:
            n_samples_val = self._num_of_samples[validation_dataset]
            XY_val = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in self._data[validation_dataset].items()}
            val_indexes = self._get_batch_indexes(validation_dataset, prediction_samples)
        if test_dataset in datasets:
            n_samples_test = self._num_of_samples[test_dataset]
            XY_test = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in self._data[test_dataset].items()}
            test_indexes = self._get_batch_indexes(test_dataset, prediction_samples)
        check(n_samples_train > 0, ValueError, f'The number of train samples {n_samples_train} must be greater than 0.')
        if prediction_samples > 0:
            train_indexes = train_indexes[:-prediction_samples]
            if validation_dataset in datasets:
                val_indexes = val_indexes[:-prediction_samples]
            if test_dataset in datasets:
                test_indexes = test_indexes[:-prediction_samples]

        return XY_train, XY_val, XY_test, n_samples_train, n_samples_val, n_samples_test, train_indexes, val_indexes, test_indexes

    def _setup_dataset(self, train_dataset, validation_dataset, test_dataset, dataset, splits, prediction_samples):
        if train_dataset is None and dataset is None:
            ## TODO: change to all the datasets loaded
            dataset = list(self._data.keys())[0]
        if train_dataset:  ## Use each dataset
            return self.__get_data(train_dataset, validation_dataset, test_dataset, prediction_samples)
        else:  ## use the splits
            return self.__split_dataset(dataset, splits, prediction_samples)

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
            for ind, key in enumerate(self._model_def['Minimizers'].keys()):
                loss = sum(horizon_losses[ind]) / (prediction_samples + 1)
                aux_losses[ind][batch_val] = loss.item()
                if total_losses is not None:
                    total_losses[key].append(loss.detach().numpy())
                total_loss += loss

            ## Gradient Step
            if optimizer:
                total_loss.backward()  ## Backpropagate the error
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
            check(output in self._model_def['Outputs'], ValueError, f'the tag {output} is not an output of the network')
            log.warning(f'Recurrent train: closing the loop between the the input ports {input} and the output ports {output} for {prediction_samples} samples')
        ## Connect information
        for connect_in, connect_out in connect.items():
            check(connect_in in self._model_def['Inputs'], ValueError, f'the tag {connect_in} is not an input variable.')
            check(connect_out in self._model_def['Outputs'], ValueError, f'the tag {connect_out} is not an output of the network')
            log.warning(f'Recurrent train: connecting the input ports {connect_in} with output ports {connect_out} for {prediction_samples} samples')
        ## Disable recurrent training if there are no recurrent variables
        if len(connect|closed_loop|self._model_def.recurrentInputs()) == 0:
            if prediction_samples >= 0:
                log.warning(f"The value of the prediction_samples={prediction_samples} but the network has no recurrent variables.")
            prediction_samples = -1
        return prediction_samples

    @enforce_types
    def resetStates(self, states:set={}, *, batch:int=1) -> None:
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

