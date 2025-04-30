import copy, torch, random, inspect
from copy import deepcopy

from inspect import signature
from collections.abc import Callable

from nnodely.basic.modeldef import ModelDef
from nnodely.basic.model import Model
from nnodely.basic.optimizer import Optimizer, SGD, Adam
from nnodely.basic.loss import CustomLoss
from nnodely.operators.network import Network
from nnodely.support.utils import tensor_to_list, check, TORCH_DTYPE, enforce_types, ReadOnlyDict
from nnodely.basic.relation import Stream
from nnodely.layers.output import Output

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

def to_training_params(func):
    def wrapper(*args, **kwargs):
        sig = signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)

        # Initialize training_params
        if 'training_params' not in kwargs:
            kwargs['training_params'] = {}
        else:
            kwargs['training_params'] = copy.deepcopy(kwargs['training_params'])
            if 'lr' in kwargs['training_params']:
                kwargs['training_params']['default_lr'] = kwargs['training_params']['lr']
            if 'lr_param' in kwargs['training_params']:
                kwargs['training_params']['default_lr_param'] = kwargs['training_params']['lr_param']

        # Add the default parameters that are not present in training_params
        for key, value in sig.parameters.items():
            if (sig.parameters[key].default != None and key not in kwargs['training_params']) and key not in {'self', 'training_params'}:
                if key == 'lr':
                    kwargs['training_params']['default_lr'] = value.default
                elif key == 'lr_param':
                    kwargs['training_params']['default_lr_param'] = value.default
                else:
                    kwargs['training_params'][key] = value.default

        if 'lr' in kwargs['training_params']:
            del kwargs['training_params']['lr']
        if 'lr_param' in kwargs['training_params']:
            del kwargs['training_params']['lr_param']
        # # Initialize optimizer defaults
        # if 'optimizer_defaults' not in kwargs:
        #     kwargs['training_params']['optimizer_defaults'] = {'lr' :kwargs['training_params']['lr']}
        # else:
        #     kwargs['training_params']['optimizer_defaults'] = bound_args.arguments['optimizer_defaults'].copy()

        for key, value in bound_args.arguments.items():
            if key in kwargs and key not in {'self', 'training_params'}:
                kwargs['training_params'][key] = copy.deepcopy(value)

        return func(*args, **kwargs)
    return wrapper

class Trainer(Network):
    def __init__(self, log_internal:bool = False):
        check(type(self) is not Trainer, TypeError, "Trainer class cannot be instantiated directly")
        super().__init__()

        # Training Parameters
        self.__standard_train_parameters = {
            'models' : None,
            'train_dataset' : None, 'validation_dataset' : None, 'test_dataset' : None, 'splits' : [70, 20, 10],
            'closed_loop' : {}, 'connect' : {}, 'step' : 0, 'prediction_samples' : 0,
            'shuffle_data' : True,
            'early_stopping' : None, 'early_stopping_params' : {},
            'select_model' : 'last', 'select_model_params' : {},
            'minimize_gain' : {},
            'num_of_epochs': 100,
            'train_batch_size' : 128, 'val_batch_size' : None, 'test_batch_size' : None,
            'optimizer' : 'Adam',
            'lr' : 0.001, 'lr_param' : {},
            'optimizer_params' : [], 'add_optimizer_params' : [],
            'optimizer_defaults' : {}, 'add_optimizer_defaults' : {}
        }

        # Training Losses
        self.__loss_functions = {}

        # Optimizer
        self.__optimizer = None

        # Save internal
        self.__log_internal = log_internal
        if self.__log_internal == True:
            self.__internals = {}

    @property
    def internals(self):
        return ReadOnlyDict(self.__internals)

    def __save_internal(self, key, value):
        self.__internals[key] = tensor_to_list(value)

    # def __get_train_parametersOLD(self, training_params):
    #     run_train_parameters = copy.deepcopy(self.__standard_train_parameters)
    #     if training_params is None:
    #         return run_train_parameters
    #     for key, value in training_params.items():
    #         check(key in run_train_parameters, KeyError, f"The param {key} is not exist as standard parameters")
    #         run_train_parameters[key] = value
    #     return run_train_parameters
    #
    # def __get_parameter(self, **parameter):
    #     assert len(parameter) == 1
    #     name = list(parameter.keys())[0]
    #     self.run_training_params[name] = parameter[name] if parameter[name] is not None else self.run_training_params[name]
    #     return self.run_training_params[name]

    def __get_parameter_NEW(self, training_params, name):
        if name not in training_params:
            training_params[name] = self.__standard_train_parameters[name]
        return training_params[name]

    def __get_batch_sizes(self, training_params):
        ## Check if the batch_size can be used for the current dataset, otherwise set the batch_size to the maximum value
        if training_params['recurrent_train']:
            if training_params['train_batch_size'] > training_params['n_samples_train']:
                training_params['train_batch_size'] = training_params['n_samples_train'] - \
                                                               training_params['prediction_samples']
            if training_params['val_batch_size'] is None or training_params['val_batch_size'] > \
                    training_params['n_samples_val']:
                training_params['val_batch_size'] = max(0, training_params['n_samples_val'] -
                                                                 training_params['prediction_samples'])
            if training_params['test_batch_size'] is None or training_params['test_batch_size'] > \
                    training_params['n_samples_test']:
                training_params['test_batch_size'] = max(0, training_params['n_samples_test'] -
                                                                  training_params['prediction_samples'])
        else:
            if training_params['train_batch_size'] > training_params['n_samples_train']:
                training_params['train_batch_size'] = training_params['n_samples_train']
            if training_params['val_batch_size'] is None or training_params['val_batch_size'] > \
                    training_params['n_samples_val']:
                training_params['val_batch_size'] = training_params['n_samples_val']
            if training_params['test_batch_size'] is None or training_params['test_batch_size'] > \
                    training_params['n_samples_test']:
                training_params['test_batch_size'] = training_params['n_samples_test']

        check(training_params['train_batch_size'] > 0, ValueError,
              f'The auto train_batch_size ({training_params["train_batch_size"]}) = n_samples_train ({training_params["n_samples_train"]}) - prediction_samples ({training_params["prediction_samples"]}), must be greater than 0.')

    def __inizilize_optimizer(self, training_params, optimizer, optimizer_params, optimizer_defaults, add_optimizer_params,
                              add_optimizer_defaults, models, lr, lr_param):
        # Get optimizer and initialization parameters
        optimizer = copy.deepcopy(training_params['optimizer'])
        optimizer_params = training_params['optimizer_params']
        #optimizer_defaults = training_params['optimizer_defaults']
        add_optimizer_params = training_params['add_optimizer_params']
        add_optimizer_defaults = training_params['add_optimizer_defaults']
        models = training_params['models']

        ## Get parameter to be trained
        json_models = []
        if 'Models' in self._model_def:
            json_models = list(self._model_def['Models'].keys()) if type(self._model_def['Models']) is dict else [
                self._model_def['Models']]
        if models == 'ALL':
            models = json_models
        self.run_training_params['models'] = models
        params_to_train = set()
        if isinstance(models, str):
            models = [models]
        for model in models:
            check(model in json_models, ValueError, f'The model {model} is not in the model definition')
            if type(self._model_def['Models']) is dict:
                params_to_train |= set(self._model_def['Models'][model]['Parameters'])
            else:
                params_to_train |= set(self._model_def['Parameters'].keys())

        # Get the optimizer
        if type(optimizer) is str:
            if optimizer == 'SGD':
                optimizer = SGD({}, [])
            elif optimizer == 'Adam':
                optimizer = Adam({}, [])
        else:
            check(issubclass(type(optimizer), Optimizer), TypeError,
                  "The optimizer must be an Optimizer or str")

        optimizer.set_params_to_train(self._model.all_parameters, params_to_train)

        optimizer.add_defaults('lr', training_params['default_lr'])
        optimizer.add_option_to_params('lr', training_params['default_lr_param'])

        if 'optimizer_defaults' in training_params:
            optimizer.set_defaults(training_params['optimizer_defaults'])
        if optimizer_params != []:
            optimizer.set_params(optimizer_params)

        for key, value in add_optimizer_defaults.items():
            optimizer.add_defaults(key, value)

        add_optimizer_params = optimizer.unfold(add_optimizer_params)
        for param in add_optimizer_params:
            par = param['params']
            del param['params']
            for key, value in param.items():
                optimizer.add_option_to_params(key, {par: value})

        # Modify the parameter
        if 'lr' in training_params:
            optimizer.add_defaults('lr', training_params['lr'])
        if 'lr_param' in training_params:
            optimizer.add_option_to_params('lr', training_params['lr_param'])

        return optimizer


    def __get_batch_indexes(self, dataset_name, n_samples, prediction_samples, batch_size, step, type='train'):
        available_samples = n_samples - prediction_samples
        batch_indexes = list(range(available_samples))
        if dataset_name in self._multifile.keys():
            if type == 'train':
                start_idx, end_idx = 0, n_samples
            elif type == 'val':
                start_idx, end_idx = self.run_training_params['n_samples_train'], self.run_training_params[
                                                                                      'n_samples_train'] + n_samples
            elif type == 'test':
                start_idx, end_idx = self.run_training_params['n_samples_train'] + self.run_training_params[
                    'n_samples_val'], self.run_training_params['n_samples_train'] + self.run_training_params[
                                         'n_samples_val'] + n_samples

            forbidden_idxs = []
            for i in self._multifile[dataset_name]:
                if i < end_idx and i > start_idx:
                    forbidden_idxs.extend(range(i - prediction_samples, i, 1))
            batch_indexes = [idx for idx in batch_indexes if idx not in forbidden_idxs]

        ## Clip the step
        clipped_step = copy.deepcopy(step)
        if clipped_step < 0:  ## clip the step to zero
            log.warning(f"The step is negative ({clipped_step}). The step is set to zero.", stacklevel=5)
            clipped_step = 0
        if clipped_step > (len(batch_indexes) - batch_size):  ## Clip the step to the maximum number of samples
            log.warning(
                f"The step ({clipped_step}) is greater than the number of available samples ({len(batch_indexes) - batch_size}). The step is set to the maximum number.",
                stacklevel=5)
            clipped_step = len(batch_indexes) - batch_size
        ## Loss vector
        check((batch_size + clipped_step) > 0, ValueError,
              f"The sum of batch_size={batch_size} and the step={clipped_step} must be greater than 0.")

        return batch_indexes, clipped_step

    def __recurrentTrain(self, data, batch_indexes, batch_size, loss_gains, prediction_samples,
                         step, non_mandatory_inputs, mandatory_inputs, shuffle=False, train=True):
        indexes = copy.deepcopy(batch_indexes)
        json_inputs = self._model_def['Inputs']
        aux_losses = torch.zeros(
            [len(self._model_def['Minimizers']), round((len(indexes) + step) / (batch_size + step))])
        X = {}
        batch_val = 0
        while len(indexes) >= batch_size:
            idxs = random.sample(indexes, batch_size) if shuffle else indexes[:batch_size]
            for num in idxs:
                indexes.remove(num)
            if step > 0:
                if len(indexes) >= step:
                    step_idxs = random.sample(indexes, step) if shuffle else indexes[:step]
                    for num in step_idxs:
                        indexes.remove(num)
                else:
                    indexes = []
            if train:
                self.__optimizer.zero_grad()  ## Reset the gradient
            ## Reset
            init_states = []
            horizon_losses = {ind: [] for ind in range(len(self._model_def['Minimizers']))}
            for key in non_mandatory_inputs:
                if key in data.keys(): ## with data
                    X[key] = data[key][idxs]
                else:  ## with zeros
                    window_size = self._input_n_samples[key]
                    dim = json_inputs[key]['dim']
                    if 'type' in json_inputs[key]:
                        X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=TORCH_DTYPE, requires_grad=True)
                    else:
                        X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=TORCH_DTYPE,
                                             requires_grad=False)
                    self._states[key] = X[key]

                    # if 'init' in json_inputs[key].keys(): ## with init relation
                    #     self._model.connect_update[key] = json_inputs[key]['init']
                    #     init_states.append(key)


            for horizon_idx in range(prediction_samples + 1):
                ## Get data
                for key in mandatory_inputs:
                    X[key] = data[key][[idx + horizon_idx for idx in idxs]]
                ## Forward pass
                out, minimize_out, out_closed_loop, out_connect = self._model(X)

                if self.__log_internal and train:
                    #assert (check_gradient_operations(self._states) == 0)
                    #assert (check_gradient_operations(data) == 0)
                    internals_dict = {'XY': tensor_to_list(X), 'out': out, 'param': self._model.all_parameters,
                                      'closedLoop': self._model.closed_loop_update, 'connect': self._model.connect_update}

                ## Loss Calculation
                for ind, (key, value) in enumerate(self._model_def['Minimizers'].items()):
                    loss = self.__loss_functions[key](minimize_out[value['A']], minimize_out[value['B']])
                    loss = (loss * loss_gains[
                        key]) if key in loss_gains.keys() else loss  ## Multiply by the gain if necessary
                    horizon_losses[ind].append(loss)

                ## Update
                self._updateState(X, out_closed_loop, out_connect)
                ## remove initialization in closed_loop
                if init_states:
                    for key in init_states:
                        del self._model.connect_update[key]
                    init_states = []

                if self.__log_internal and train:
                    internals_dict['state'] = self._states
                    self.__save_internal('inout_' + str(batch_val) + '_' + str(horizon_idx), internals_dict)

            ## Calculate the total loss
            total_loss = 0
            for ind in range(len(self._model_def['Minimizers'])):
                loss = sum(horizon_losses[ind]) / (prediction_samples + 1)
                aux_losses[ind][batch_val] = loss.item()
                total_loss += loss

            ## Gradient Step
            if train:
                total_loss.backward()  ## Backpropagate the error
                self.__optimizer.step()
                self.visualizer.showWeightsInTrain(batch=batch_val)
            batch_val += 1

        ## return the losses
        return aux_losses

    def __Train(self, data, n_samples, batch_size, loss_gains, shuffle=True, train=True):
        check((n_samples - batch_size + 1) > 0, ValueError,
              f"The number of available sample are (n_samples_train - train_batch_size + 1) = {n_samples - batch_size + 1}.")
        if shuffle:
            randomize = torch.randperm(n_samples)
            data = {key: val[randomize] for key, val in data.items()}
        ## Initialize the train losses vector
        aux_losses = torch.zeros([len(self._model_def['Minimizers']), n_samples // batch_size])
        for idx in range(0, (n_samples - batch_size + 1), batch_size):
            ## Build the input tensor
            XY = {key: val[idx:idx + batch_size] for key, val in data.items()}
            ## Reset gradient
            if train:
                self.__optimizer.zero_grad()
            ## Model Forward
            _, minimize_out, _, _ = self._model(XY)  ## Forward pass
            ## Loss Calculation
            total_loss = 0
            for ind, (key, value) in enumerate(self._model_def['Minimizers'].items()):
                loss = self.__loss_functions[key](minimize_out[value['A']], minimize_out[value['B']])
                loss = (loss * loss_gains[
                    key]) if key in loss_gains.keys() else loss  ## Multiply by the gain if necessary
                aux_losses[ind][idx // batch_size] = loss.item()
                total_loss += loss
            ## Gradient step
            if train:
                total_loss.backward()
                self.__optimizer.step()
                self.visualizer.showWeightsInTrain(batch=idx // batch_size)

        ## return the losses
        return aux_losses

    @enforce_types
    def addMinimize(self, name:str, streamA:Stream|Output, streamB:Stream|Output, loss_function:str='mse') -> None:
        """
        Adds a minimize loss function to the model.

        Parameters
        ----------
        name : str
            The name of the cost function.
        streamA : Stream
            The first relation stream for the minimize operation.
        streamB : Stream
            The second relation stream for the minimize operation.
        loss_function : str, optional
            The loss function to use from the ones provided. Default is 'mse'.

        Example
        -------
        Example usage:
            >>> model.addMinimize('minimize_op', streamA, streamB, loss_function='mse')
        """
        self._model_def.addMinimize(name, streamA, streamB, loss_function)
        self.visualizer.showaddMinimize(name)

    @enforce_types
    def removeMinimize(self, name_list:list|str) -> None:
        """
        Removes minimize loss functions using the given list of names.

        Parameters
        ----------
        name_list : list of str
            The list of minimize operation names to remove.

        Example
        -------
        Example usage:
            >>> model.removeMinimize(['minimize_op1', 'minimize_op2'])
        """
        self._model_def.removeMinimize(name_list)


    def __setup_reecurrent_train(self, training_params, closed_loop, connect, step, prediction_samples):
        ## Get information for recurrent train
        #TODO support also prediction_samples = None as for __call__ disconnectboth closed_loop and connect
        check(prediction_samples >= 0, KeyError, 'The sample horizon must be positive!')
        recurrent_train = True

        ## Closed loop information
        all_closed_loop = training_params['closed_loop'] | self._model_def._input_closed_loop
        for input, output in all_closed_loop.items():
            check(input in self._model_def['Inputs'], ValueError, f'the tag {input} is not an input variable.')
            check(output in self._model_def['Outputs'], ValueError,
                  f'the tag {output} is not an output of the network')
            log.warning(
                f'Recurrent train: closing the loop between the the input ports {input} and the output ports {output} for {prediction_samples} samples')

        ## Connect information
        all_connect = training_params['connect']  | self._model_def._input_connect
        for connect_in, connect_out in all_connect.items():
            check(connect_in in self._model_def['Inputs'], ValueError,
                  f'the tag {connect_in} is not an input variable.')
            check(connect_out in self._model_def['Outputs'], ValueError,
                  f'the tag {connect_out} is not an output of the network')
            log.warning(
                f'Recurrent train: connecting the input ports {connect_in} with output ports {connect_out} for {prediction_samples} samples')

        ## Disable recurrent training if there are no recurrent variables
        if len(all_connect|all_closed_loop|self._model_def.recurrentInputs()) == 0 or prediction_samples == None:
            if prediction_samples != None:
                log.warning(
                    f"The value of the prediction_samples={prediction_samples} but the network has no recurrent variables.")
            recurrent_train = False

        training_params['recurrent_train'] = recurrent_train
        training_params['all_connect'] = all_connect
        training_params['all_closed_loop'] = all_closed_loop

    def __setup_dataset(self, training_params, train_dataset, validation_dataset, test_dataset, splits):
        # TODO manage multiple datasets
        if 'train_dataset' not in training_params:  ## If we use all datasets with the splits
            #TODO Move to preliminary checks?
            splits = self.__get_parameter_NEW(training_params, 'splits')
            check(len(splits) == 3, ValueError,
                  '3 elements must be inserted for the dataset split in training, validation and test')
            check(sum(splits) == 100, ValueError, 'Training, Validation and Test splits must sum up to 100.')
            check(splits[0] > 0, ValueError, 'The training split cannot be zero.')

            ## Get the dataset name
            dataset_file = list(self._data.keys())[0]  ## take the dataset name
            train_dataset_file = validation_dataset_file = test_dataset_file = dataset_file

            ## Collect the split sizes
            train_size = splits[0] / 100.0
            val_size = splits[1] / 100.0
            test_size = 1 - (train_size + val_size)
            num_of_samples = self._num_of_samples[dataset_file]
            n_samples_train = round(num_of_samples * train_size)
            if splits[1] == 0:
                n_samples_test = num_of_samples - n_samples_train
                n_samples_val = 0
            else:
                n_samples_test = round(num_of_samples * test_size)
                n_samples_val = num_of_samples - n_samples_train - n_samples_test

            ## Split into train, validation and test
            XY_train, XY_val, XY_test = {}, {}, {}
            for key, samples in self._data[dataset_file].items():
                if val_size == 0.0 and test_size == 0.0:  ## we have only training set
                    XY_train[key] = torch.from_numpy(samples).to(TORCH_DTYPE)
                elif val_size == 0.0 and test_size != 0.0:  ## we have only training and test set
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                    XY_test[key] = torch.from_numpy(samples[n_samples_train:]).to(TORCH_DTYPE)
                elif val_size != 0.0 and test_size == 0.0:  ## we have only training and validation set
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                    XY_val[key] = torch.from_numpy(samples[n_samples_train:]).to(TORCH_DTYPE)
                else:  ## we have training, validation and test set
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                    XY_val[key] = torch.from_numpy(samples[n_samples_train:-n_samples_test]).to(TORCH_DTYPE)
                    XY_test[key] = torch.from_numpy(samples[n_samples_train + n_samples_val:]).to(TORCH_DTYPE)

            ## Set name for resultsAnalysis
            train_dataset = f"train_{dataset_file}_{train_size:0.2f}"
            validation_dataset = f"validation_{dataset_file}_{val_size:0.2f}"
            test_dataset = f"test_{dataset_file}_{test_size:0.2f}"
        else:  ## Multi-Dataset
            ## Get the names of the datasets
            datasets = list(self._data.keys())
            train_dataset_file, validation_dataset_file, test_dataset_file = train_dataset, validation_dataset, test_dataset

            ## Collect the number of samples for each dataset
            n_samples_train, n_samples_val, n_samples_test = 0, 0, 0

            check(train_dataset in datasets, KeyError, f'{train_dataset} Not Loaded!')
            if validation_dataset is not None and validation_dataset not in datasets:
                log.warning(
                    f'Validation Dataset [{validation_dataset}] Not Loaded. The training will continue without validation')
            if test_dataset is not None and test_dataset not in datasets:
                log.warning(f'Test Dataset [{test_dataset}] Not Loaded. The training will continue without test')

            ## Split into train, validation and test
            XY_train, XY_val, XY_test = {}, {}, {}
            n_samples_train = self._num_of_samples[train_dataset]
            XY_train = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in self._data[train_dataset].items()}
            if validation_dataset in datasets:
                n_samples_val = self._num_of_samples[validation_dataset]
                XY_val = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in
                          self._data[validation_dataset].items()}
            if test_dataset in datasets:
                n_samples_test = self._num_of_samples[test_dataset]
                XY_test = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in self._data[test_dataset].items()}

        for key in XY_train.keys():
            assert n_samples_train == XY_train[key].shape[
                0], f'The number of train samples {n_samples_train}!={XY_train[key].shape[0]} not compliant.'
            if key in XY_val:
                assert n_samples_val == XY_val[key].shape[
                    0], f'The number of val samples {n_samples_val}!={XY_val[key].shape[0]} not compliant.'
            if key in XY_test:
                assert n_samples_test == XY_test[key].shape[
                    0], f'The number of test samples {n_samples_test}!={XY_test[key].shape[0]} not compliant.'

        self.run_training_params['n_samples_train'] = n_samples_train
        self.run_training_params['n_samples_val'] = n_samples_val
        self.run_training_params['n_samples_test'] = n_samples_test
        self.run_training_params['train_dataset_file'] = train_dataset_file
        self.run_training_params['validation_dataset_file'] = validation_dataset_file
        self.run_training_params['test_dataset_file'] = test_dataset_file
        self.run_training_params['train_dataset'] = train_dataset
        self.run_training_params['validation_dataset'] = validation_dataset
        self.run_training_params['test_dataset'] = test_dataset
        self.run_training_params['XY_train'] = XY_train
        self.run_training_params['XY_val'] = XY_val
        self.run_training_params['XY_test'] = XY_test

        training_params['n_samples_train'] = n_samples_train
        training_params['n_samples_val'] = n_samples_val
        training_params['n_samples_test'] = n_samples_test
        training_params['train_dataset_file'] = train_dataset_file
        training_params['validation_dataset_file'] = validation_dataset_file
        training_params['test_dataset_file'] = test_dataset_file
        training_params['train_dataset'] = train_dataset
        training_params['validation_dataset'] = validation_dataset
        training_params['test_dataset'] = test_dataset
        training_params['XY_train'] = XY_train
        training_params['XY_val'] = XY_val
        training_params['XY_test'] = XY_test

        assert n_samples_train > 0, f'There are {n_samples_train} samples for training.'

    def __setup_batch_indexes(self, train_dataset_file, n_samples_train, train_batch_size,
                                    validation_dataset_file, n_samples_val, val_batch_size,
                              recurrent_train, all_closed_loop, all_connect, prediction_samples, step):
        if recurrent_train:
            list_of_batch_indexes = range(0, (n_samples_train - train_batch_size - prediction_samples + 1),
                                          (train_batch_size + step))
            check(n_samples_train - train_batch_size - prediction_samples + 1 > 0, ValueError,
                  f"The number of available sample are (n_samples_train ({n_samples_train}) - train_batch_size ({train_batch_size}) - prediction_samples ({prediction_samples}) + 1) = {n_samples_train - train_batch_size - prediction_samples + 1}.")
            update_per_epochs = (n_samples_train - train_batch_size - prediction_samples + 1) // (
                        train_batch_size + step) + 1
            unused_samples = n_samples_train - list_of_batch_indexes[-1] - train_batch_size - prediction_samples

            model_inputs = list(self._model_def['Inputs'].keys())
            non_mandatory_inputs = list(all_closed_loop.keys()) + list(all_connect.keys()) +  list(self._model_def.recurrentInputs().keys())
            mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))

            list_of_batch_indexes_train, train_step = self.__get_batch_indexes(train_dataset_file, n_samples_train,
                                                                               prediction_samples, train_batch_size,
                                                                               step, type='train')
            if n_samples_val > 0:
                list_of_batch_indexes_val, val_step = self.__get_batch_indexes(validation_dataset_file, n_samples_val,
                                                                               prediction_samples, val_batch_size, step,
                                                                               type='val')
                self.run_training_params['list_of_batch_indexes_val'] = list_of_batch_indexes_val
                self.run_training_params['val_step'] = val_step

            self.run_training_params['list_of_batch_indexes_train'] = list_of_batch_indexes_train
            self.run_training_params['train_step'] = train_step
            self.run_training_params['mandatory_inputs'] = mandatory_inputs
            self.run_training_params['non_mandatory_inputs'] = non_mandatory_inputs
        else:
            update_per_epochs = (n_samples_train - train_batch_size) // train_batch_size + 1
            unused_samples = n_samples_train - update_per_epochs * train_batch_size

        self.run_training_params['update_per_epochs'] = update_per_epochs
        self.run_training_params['unused_samples'] = unused_samples

    def __preliminary_checks(self):
        check(self._data_loaded, RuntimeError, 'There is no _data loaded! The Training will stop.')
        check('Models' in self._model_def.getJson(), RuntimeError,
              'There are no models to train. Load a model using the addModel function.')
        check(list(self._model.parameters()), RuntimeError,
              'There are no models with learnable parameters! The Training will stop.')

    def __get_train_parameters(self, training_params, early_stopping, early_stopping_params, shuffle_data, num_of_epochs):
        self.run_training_params = {} #copy.deepcopy(self.__standard_train_parameters)
        for key, value in training_params.items():
            #check(key in self.run_training_params, KeyError, f"The param {key} is not exist as standard parameters")
            self.run_training_params[key] = value

        ## Get early stopping
        #early_stopping = self.__get_parameter(early_stopping=early_stopping)
        #self.run_training_params['early_stopping'] = early_stopping
        #TODO Move on the last part

        #early_stopping_params = self.__get_parameter(early_stopping_params=early_stopping_params)

    @to_training_params # Move all user parameters to the dict training_params
    @enforce_types
    def trainModel(self, *,
                   models: str | list = 'ALL',
                   #TODO verify the name validation in all user names
                   train_dataset: str | None = None, validation_dataset: str | None = None, test_dataset: str | None = None,
                   splits: list | None = None, #TODO add dataset variable used when there is only one dataset?
                   # TODO prediction_samples = None as default parameter?
                   closed_loop: dict = {}, connect: dict = {}, step: int = 0, prediction_samples: int | None = 0,
                   shuffle_data: bool = True,
                   early_stopping: Callable | None = None, early_stopping_params: dict | None = {},
                   select_model: Callable | str = 'last', select_model_params: dict = {},
                   minimize_gain: dict = {},
                   num_of_epochs: int = 100,
                   train_batch_size: int = 128, val_batch_size: int = 128, test_batch_size: int = 128,
                   optimizer: str | Optimizer = 'Adam',
                   lr: int | float =  0.001, lr_param: dict = {},
                   optimizer_defaults: dict | None = None, optimizer_params: list = [],
                   add_optimizer_defaults: dict = {}, add_optimizer_params: list = [],
                   training_params: dict | None = None
                   ) -> None:
        """
        Trains the model using the provided datasets and parameters.

        Parameters
        ----------
        models : list or None, optional
            A list of models to train. Default is None.
        train_dataset : str or None, optional
            The name of the training dataset. Default is None.
        validation_dataset : str or None, optional
            The name of the validation dataset. Default is None.
        test_dataset : str or None, optional
            The name of the test dataset. Default is None.
        splits : list or None, optional
            A list of 3 elements specifying the percentage of splits for training, validation, and testing. The three elements must sum up to 100!
            The parameter splits is only used when there is only 1 dataset loaded. Default is None.
        closed_loop : dict or None, optional
            A dictionary specifying closed loop connections. The keys are input names and the values are output names. Default is None.
        connect : dict or None, optional
            A dictionary specifying connections. The keys are input names and the values are output names. Default is None.
        step : int or None, optional
            The step size for training. A big value will result in less data used for each epochs and a faster train. Default is None.
        prediction_samples : int or None, optional
            The size of the prediction horizon. Number of samples at each recurrent window Default is None.
        shuffle_data : bool or None, optional
            Whether to shuffle the data during training. Default is None.
        early_stopping : Callable or None, optional
            A callable for early stopping. Default is None.
        early_stopping_params : dict or None, optional
            A dictionary of parameters for early stopping. Default is None.
        select_model : Callable or None, optional
            A callable for selecting the best model. Default is None.
        select_model_params : dict or None, optional
            A dictionary of parameters for selecting the best model. Default is None.
        minimize_gain : dict or None, optional
            A dictionary specifying the gain for each minimization loss function. Default is None.
        num_of_epochs : int or None, optional
            The number of epochs to train the model. Default is None.
        train_batch_size : int or None, optional
            The batch size for training. Default is None.
        val_batch_size : int or None, optional
            The batch size for validation. Default is None.
        test_batch_size : int or None, optional
            The batch size for testing. Default is None.
        optimizer : Optimizer or None, optional
            The optimizer to use for training. Default is None.
        lr : float or None, optional
            The learning rate. Default is None.
        lr_param : dict or None, optional
            A dictionary of learning rate parameters. Default is None.
        optimizer_params : list or dict or None, optional
            A dictionary of optimizer parameters. Default is None.
        optimizer_defaults : dict or None, optional
            A dictionary of default optimizer settings. Default is None.
        training_params : dict or None, optional
            A dictionary of training parameters. Default is None.
        add_optimizer_params : list or None, optional
            Additional optimizer parameters. Default is None.
        add_optimizer_defaults : dict or None, optional
            Additional default optimizer settings. Default is None.

        Raises
        ------
        RuntimeError
            If no data is loaded or if there are no modules with learnable parameters.
        KeyError
            If the sample horizon is not positive.
        ValueError
            If an input or output variable is not in the model definition.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/training.ipynb
            :alt: Open in Colab

        Example - basic feed-forward training:
            >>> x = Input('x')
            >>> F = Input('F')

            >>> xk1 = Output('x[k+1]', Fir()(x.tw(0.2))+Fir()(F.last()))

            >>> mass_spring_damper = Modely(seed=0)
            >>> mass_spring_damper.addModel('xk1',xk1)
            >>> mass_spring_damper.neuralizeModel(sample_time = 0.05)

            >>> data_struct = ['time','x','dx','F']
            >>> data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','data')
            >>> mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

            >>> params = {'num_of_epochs': 100,'train_batch_size': 128,'lr':0.001}
            >>> mass_spring_damper.trainModel(splits=[70,20,10], training_params = params)

        Example - recurrent training:
            >>> x = Input('x')
            >>> F = Input('F')

            >>> xk1 = Output('x[k+1]', Fir()(x.tw(0.2))+Fir()(F.last()))

            >>> mass_spring_damper = Modely(seed=0)
            >>> mass_spring_damper.addModel('xk1',xk1)
            >>> mass_spring_damper.addClosedLoop(xk1, x)
            >>> mass_spring_damper.neuralizeModel(sample_time = 0.05)

            >>> data_struct = ['time','x','dx','F']
            >>> data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','data')
            >>> mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

            >>> params = {'num_of_epochs': 100,'train_batch_size': 128,'lr':0.001}
            >>> mass_spring_damper.trainModel(splits=[70,20,10], prediction_samples=10, training_params = params)
        """
        self.__preliminary_checks()
        self.__get_train_parameters(training_params, early_stopping, early_stopping_params, shuffle_data, num_of_epochs)
        num_of_epochs = training_params['num_of_epochs']

        ## Enable log internal for debugging
        if self.__log_internal:
            self.__internals = {}

        ## Get the recurrent variables
        self.__setup_reecurrent_train(training_params, closed_loop, connect, step, prediction_samples)
        all_closed_loop = training_params['all_closed_loop']
        all_connect = training_params['all_connect']
        step = training_params['step']
        prediction_samples = training_params['prediction_samples']
        recurrent_train = self.run_training_params['recurrent_train'] = training_params['recurrent_train']

        ## Get the dataset name
        self.__setup_dataset(training_params, train_dataset, validation_dataset, test_dataset, splits)
        n_samples_train = self.run_training_params['n_samples_train']
        n_samples_val = self.run_training_params['n_samples_val']
        n_samples_test = self.run_training_params['n_samples_test']
        XY_train = self.run_training_params['XY_train']
        XY_val = self.run_training_params['XY_val']
        XY_test = self.run_training_params['XY_test']
        train_dataset = self.run_training_params['train_dataset']
        validation_dataset = self.run_training_params['validation_dataset']
        test_dataset = self.run_training_params['test_dataset']
        train_dataset_file = self.run_training_params['train_dataset_file']
        validation_dataset_file = self.run_training_params['validation_dataset_file']
        test_dataset_file = self.run_training_params['test_dataset_file']

        ## Get the batch size
        self.__get_batch_sizes(training_params)
        train_batch_size = self.run_training_params['train_batch_size']  = training_params['train_batch_size']
        val_batch_size =   self.run_training_params['val_batch_size'] = training_params['val_batch_size']
        test_batch_size =  self.run_training_params['test_batch_size'] = training_params['test_batch_size']

        ## Define the optimizer
        optimizer = self.__inizilize_optimizer(training_params, optimizer, optimizer_params, optimizer_defaults, add_optimizer_params,
                                               add_optimizer_defaults, models, lr, lr_param)
        self.run_training_params['optimizer'] = optimizer.name
        self.run_training_params['optimizer_params'] = optimizer.optimizer_params
        self.run_training_params['optimizer_defaults'] = optimizer.optimizer_defaults
        self.__optimizer = optimizer.get_torch_optimizer()

        ## Define the loss functions
        minimize_gain = training_params['minimize_gain']
        self.run_training_params['minimizers'] = {}
        for name, values in self._model_def['Minimizers'].items():
            self.__loss_functions[name] = CustomLoss(values['loss'])
            self.run_training_params['minimizers'][name] = {}
            self.run_training_params['minimizers'][name]['A'] = values['A']
            self.run_training_params['minimizers'][name]['B'] = values['B']
            self.run_training_params['minimizers'][name]['loss'] = values['loss']
            if name in minimize_gain:
                self.run_training_params['minimizers'][name]['gain'] = minimize_gain[name]

        if 'early_stopping' in training_params:
            self.run_training_params['early_stopping'] = early_stopping.__name__

        ## Clean the dict of the training parameter
        del self.run_training_params['minimize_gain']
        #del self.run_training_params['lr']
        #del self.run_training_params['lr_param']
        if not recurrent_train:
            del self.run_training_params['connect']
            del self.run_training_params['closed_loop']
            del self.run_training_params['step']
            del self.run_training_params['prediction_samples']
        if early_stopping is None:
            del self.run_training_params['early_stopping_params']

        ## Create the train, validation and test loss dictionaries
        train_losses, val_losses = {}, {}
        for key in self._model_def['Minimizers'].keys():
            train_losses[key] = []
            if n_samples_val > 0:
                val_losses[key] = []

        ## Check the needed keys are in the datasets
        keys = set(self._model_def['Inputs'].keys())
        keys |= {value['A'] for value in self._model_def['Minimizers'].values()} | {value['B'] for value in
                                                                                   self._model_def[
                                                                                       'Minimizers'].values()}
        keys -= set(self._model_def['Relations'].keys())
        keys -= set(self._model_def.recurrentInputs().keys())
        keys -= set(self._model_def['Outputs'].keys())
        keys -= set(all_connect.keys())
        keys -= set(all_closed_loop.keys())
        check(set(keys).issubset(set(XY_train.keys())), KeyError,
              f"Not all the mandatory keys {keys} are present in the training dataset {set(XY_train.keys())}.")

        ## Evaluate the number of updates for epochs and the unsued samples and batch indexes
        self.__setup_batch_indexes(train_dataset_file, n_samples_train, train_batch_size,
                                    validation_dataset_file, n_samples_val, val_batch_size,
                                    recurrent_train, all_closed_loop, all_connect, prediction_samples, step)
        if recurrent_train:
            if n_samples_val > 0:
                list_of_batch_indexes_val = self.run_training_params['list_of_batch_indexes_val']
                val_step = self.run_training_params['val_step']
            list_of_batch_indexes_train = self.run_training_params['list_of_batch_indexes_train']
            train_step = self.run_training_params['train_step']
            mandatory_inputs = self.run_training_params['mandatory_inputs']
            non_mandatory_inputs = self.run_training_params['non_mandatory_inputs']

        ## Set the gradient to true if necessary
        json_inputs = self._model_def['Inputs']
        for key in json_inputs.keys():
            if 'type' in json_inputs[key]:
                if key in XY_train:
                    XY_train[key].requires_grad_(True)
                if key in XY_val:
                    XY_val[key].requires_grad_(True)
                if key in XY_test:
                    XY_test[key].requires_grad_(True)

        ## Select the model
        select_model = training_params['select_model']
        select_model_params = training_params['select_model_params']
        selected_model_def = ModelDef(self._model_def.getJson())

        ## Show the training parameters
        self.visualizer.showTrainParams()

        import time
        ## start the train timer
        start = time.time()
        self.visualizer.showStartTraining()

        ## Update with virtual states
        self._model.update(closed_loop=all_closed_loop, connect=all_connect)

        self.resetStates()  ## Reset the states

        for epoch in range(num_of_epochs):
            ## TRAIN
            self._model.train()
            if recurrent_train:
                losses = self.__recurrentTrain(XY_train, list_of_batch_indexes_train, train_batch_size, minimize_gain,
                                               prediction_samples, train_step, non_mandatory_inputs,
                                               mandatory_inputs, shuffle=shuffle_data, train=True)
            else:
                losses = self.__Train(XY_train, n_samples_train, train_batch_size, minimize_gain, shuffle=shuffle_data,
                                      train=True)
            ## save the losses
            for ind, key in enumerate(self._model_def['Minimizers'].keys()):
                train_losses[key].append(torch.mean(losses[ind]).tolist())

            if n_samples_val > 0:
                ## VALIDATION
                self._model.eval()
                if recurrent_train:
                    losses = self.__recurrentTrain(XY_val, list_of_batch_indexes_val, val_batch_size, minimize_gain,
                                                   prediction_samples, val_step, non_mandatory_inputs,
                                                   mandatory_inputs, shuffle=False, train=False)
                else:
                    losses = self.__Train(XY_val, n_samples_val, val_batch_size, minimize_gain, shuffle=False,
                                          train=False)
                ## save the losses
                for ind, key in enumerate(self._model_def['Minimizers'].keys()):
                    val_losses[key].append(torch.mean(losses[ind]).tolist())

            ## Early-stopping
            if callable(early_stopping):
                if early_stopping(train_losses, val_losses, early_stopping_params):
                    log.info(f'Stopping the training at epoch {epoch} due to early stopping.')
                    break

            if callable(select_model):
                if select_model(train_losses, val_losses, select_model_params):
                    best_model_epoch = epoch
                    selected_model_def.updateParameters(self._model)

            ## Visualize the training...
            self.visualizer.showTraining(epoch, train_losses, val_losses)
            self.visualizer.showWeightsInTrain(epoch=epoch)

        ## Save the training time
        end = time.time()
        ## Visualize the training time
        for key in self._model_def['Minimizers'].keys():
            self._training[key] = {'train': train_losses[key]}
            if n_samples_val > 0:
                self._training[key]['val'] = val_losses[key]
        self.visualizer.showEndTraining(num_of_epochs - 1, train_losses, val_losses)
        self.visualizer.showTrainingTime(end - start)

        ## Select the model
        if callable(select_model):
            log.info(f'Selected the model at the epoch {best_model_epoch + 1}.')
            self._model = Model(selected_model_def)
        else:
            log.info('The selected model is the LAST model of the training.')

        self.resultAnalysis(train_dataset, XY_train, minimize_gain, all_closed_loop, all_connect, prediction_samples, step,
                            train_batch_size)
        if self.run_training_params['n_samples_val'] > 0:
            self.resultAnalysis(validation_dataset, XY_val, minimize_gain, all_closed_loop, all_connect, prediction_samples,
                                step, val_batch_size)
        if self.run_training_params['n_samples_test'] > 0:
            self.resultAnalysis(test_dataset, XY_test, minimize_gain, all_closed_loop, all_connect, prediction_samples, step,
                                test_batch_size)

        self.visualizer.showResults()

        ## Remove virtual states
        self._removeVirtualStates(connect, closed_loop)

        ## Get trained model from torch and set the model_def
        self._model_def.updateParameters(self._model)
