import copy, torch, random

from collections.abc import Callable

from nnodely.basic.modeldef import ModelDef
from nnodely.basic.model import Model
from nnodely.basic.optimizer import Optimizer, SGD, Adam
from nnodely.basic.loss import CustomLoss
from nnodely.operators.network import Network
from nnodely.support.utils import tensor_to_list, check, TORCH_DTYPE, enforce_types, ReadOnlyDict, get_batch_size
from nnodely.basic.relation import Stream
from nnodely.layers.output import Output

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

class Trainer(Network):
    def __init__(self):
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
            'train_batch_size' : 128, 'val_batch_size' : 128, 'test_batch_size' : 128,
            'optimizer' : 'Adam',
            'lr' : 0.001, 'lr_param' : {},
            'optimizer_params' : [], 'add_optimizer_params' : [],
            'optimizer_defaults' : {}, 'add_optimizer_defaults' : {},
            'weights_function' : None
        }

        # Training Losses
        self.__loss_functions = {}

        # Optimizer
        self.__optimizer = None

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

    def __preliminary_checks(self):
        check(self._data_loaded, RuntimeError, 'There is no _data loaded! The Training will stop.')
        check('Models' in self._model_def.getJson(), RuntimeError,
              'There are no models to train. Load a model using the addModel function.')
        check(list(self._model.parameters()), RuntimeError,
              'There are no modules with learnable parameters! The Training will stop.')

    def __get_train_parameters(self, tp, training_params):
        if training_params is None:
            training_params = {}
        for key, value in training_params.items():
            check(key in self.__standard_train_parameters, KeyError, f"The param {key} is not exist as standard parameters")
            tp[key] = value

    def __get_parameter(self, tp, **parameter):
        assert len(parameter) == 1
        name = list(parameter.keys())[0]
        tp[name] = parameter[name] if parameter[name] is not None else tp[name]
        return tp[name]

    def __setup_recurrent_train(self, tp, prediction_samples, step, closed_loop, connect):
        ## Prediction samples
        step = self.__get_parameter(tp, step=step)
        prediction_samples = self.__get_parameter(tp, prediction_samples=prediction_samples)
        closed_loop = self.__get_parameter(tp, closed_loop=closed_loop)
        connect = self.__get_parameter(tp, connect=connect)
        tp['prediction_samples'] = self._setup_recurrent_variables(prediction_samples, closed_loop, connect)
        return tp['prediction_samples'], step, closed_loop, connect

    def __setup_dataset(self, tp, shuffle_data, train_dataset, validation_dataset, test_dataset, splits):
        ## Get dataset for training
        shuffle_data = self.__get_parameter(tp, shuffle_data=shuffle_data)

        # TODO manage multiple datasets
        XY_train, XY_val, XY_test = {}, {}, {}
        if train_dataset is None:  ## If we use all datasets with the splits
            splits = self.__get_parameter(tp, splits=splits)
            check(len(splits) == 3, ValueError,
                  '3 elements must be inserted for the dataset split in training, validation and test')
            check(sum(splits) == 100, ValueError, 'Training, Validation and Test splits must sum up to 100.')
            check(splits[0] > 0, ValueError, 'The training split cannot be zero.')

            ## Get the dataset name
            dataset = list(self._data.keys())[0]  ## take the dataset name
            tp['train_dataset']  = tp['validation_dataset'] = tp['test_dataset'] = \
                self.__get_parameter(tp, train_dataset=dataset)

            ## Collect the split sizes
            train_size = splits[0] / 100.0
            val_size = splits[1] / 100.0
            test_size = 1 - (train_size + val_size)
            num_of_samples = self._num_of_samples[dataset]
            n_samples_train = round(num_of_samples * train_size)
            if splits[1] == 0:
                n_samples_test = num_of_samples - n_samples_train
                n_samples_val = 0
            else:
                n_samples_test = round(num_of_samples * test_size)
                n_samples_val = num_of_samples - n_samples_train - n_samples_test

            ## Split into train, validation and test
            for key, samples in self._data[dataset].items():
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
            tp['train_dataset_name'] = f"train_{dataset}_{train_size:0.2f}"
            tp['validation_dataset_name'] = f"validation_{dataset}_{val_size:0.2f}"
            tp['test_dataset_name'] = f"test_{dataset}_{test_size:0.2f}"
        else:  ## Multi-Dataset
            ## Get the names of the datasets
            datasets = list(self._data.keys())
            train_dataset = tp['train_dataset_name'] = self.__get_parameter(tp, train_dataset=train_dataset)
            validation_dataset = tp['validation_dataset_name'] = self.__get_parameter(tp, validation_dataset=validation_dataset)
            test_dataset = tp['test_dataset_name'] = self.__get_parameter(tp, test_dataset=test_dataset)

            ## Collect the number of samples for each dataset
            n_samples_train, n_samples_val, n_samples_test = 0, 0, 0

            check(train_dataset in datasets, KeyError, f'{train_dataset} Not Loaded!')
            if validation_dataset is not None and validation_dataset not in datasets:
                log.warning(
                    f'Validation Dataset [{validation_dataset}] Not Loaded. The training will continue without validation')
            if test_dataset is not None and test_dataset not in datasets:
                log.warning(f'Test Dataset [{test_dataset}] Not Loaded. The training will continue without test')

            ## Split into train, validation and test
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

        assert n_samples_train > 0, f'There are {n_samples_train} samples for training.'
        tp['n_samples_train'], tp['n_samples_val'], tp['n_samples_test'] = n_samples_train, n_samples_val, n_samples_test
        tp['XY_train'], tp['XY_val'], tp['XY_test'] = XY_train, XY_val, XY_test
        return shuffle_data, XY_train, XY_val, XY_test, n_samples_train, n_samples_val, n_samples_test

    def __get_batch_sizes(self, tp, train_batch_size, val_batch_size, test_batch_size):
        ## Check if the batch_size can be used for the current dataset, otherwise set the batch_size to the maximum value
        self.__get_parameter(tp, train_batch_size=train_batch_size)
        self.__get_parameter(tp, val_batch_size=val_batch_size)
        self.__get_parameter(tp, test_batch_size=test_batch_size)
        tp['train_batch_size'] = get_batch_size(tp['n_samples_train'], tp['train_batch_size'], tp['prediction_samples'])
        tp['val_batch_size'] = get_batch_size(tp['n_samples_val'], tp['val_batch_size'], tp['prediction_samples'])
        tp['test_batch_size'] = get_batch_size(tp['n_samples_test'], tp['test_batch_size'], tp['prediction_samples'])

        check((tp['n_samples_train'] - tp['train_batch_size'] + 1) > 0, ValueError,
              f"The number of available sample are (n_samples_train - train_batch_size + 1)"
              f" = {tp['n_samples_train'] - tp['train_batch_size'] + 1}.")
        check(tp['train_batch_size'] > 0, ValueError,
              f'The auto train_batch_size ({tp["train_batch_size"]}) = n_samples_train ({tp["n_samples_train"]}) '
              f'- prediction_samples ({tp["prediction_samples"]}), must be greater than 0.')

        return tp['train_batch_size'], tp['val_batch_size'], tp['test_batch_size']

    def __inizilize_optimizer(self, tp, optimizer, optimizer_params, optimizer_defaults, add_optimizer_params,
                              add_optimizer_defaults, models, lr, lr_param):
        # Get optimizer and initialization parameters
        optimizer = copy.deepcopy(self.__get_parameter(tp, optimizer=optimizer))
        optimizer_params = copy.deepcopy(self.__get_parameter(tp, optimizer_params=optimizer_params))
        optimizer_defaults = copy.deepcopy(self.__get_parameter(tp, optimizer_defaults=optimizer_defaults))
        add_optimizer_params = copy.deepcopy(self.__get_parameter(tp, add_optimizer_params=add_optimizer_params))
        add_optimizer_defaults = copy.deepcopy(self.__get_parameter(tp, add_optimizer_defaults=add_optimizer_defaults))

        ## Get models
        models = self.__get_parameter(tp, models=models)
        json_models = []
        if 'Models' in self._model_def:
            json_models = list(self._model_def['Models'].keys()) if type(self._model_def['Models']) is dict else [
                self._model_def['Models']]
        if models is None:
            models = json_models
        tp['models'] = models

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

        optimizer.add_defaults('lr', tp['lr'])
        optimizer.add_option_to_params('lr', tp['lr_param'])

        if optimizer_defaults != {}:
            optimizer.set_defaults(optimizer_defaults)
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
        optimizer.add_defaults('lr', lr)
        optimizer.add_option_to_params('lr', lr_param)

        tp['optimizer'] = optimizer.name
        tp['optimizer_params'] = optimizer.optimizer_params
        tp['optimizer_defaults'] = optimizer.optimizer_defaults
        return optimizer.get_torch_optimizer()

    def __get_batch_indexes(self, tp, dataset_name, n_samples, prediction_samples, batch_size, step, type='train'):
        available_samples = n_samples - prediction_samples
        batch_indexes = list(range(available_samples))
        if dataset_name in self._multifile.keys():
            if type == 'train':
                start_idx, end_idx = 0, n_samples
            elif type == 'val':
                start_idx, end_idx = tp['n_samples_train'], tp['n_samples_train'] + n_samples
            elif type == 'test':
                start_idx, end_idx = tp['n_samples_train'] + tp['n_samples_val'], tp['n_samples_train'] + tp['n_samples_val'] + n_samples
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

    def __setup_batch_indexes(self, tp):
        # Evaluate the number of update for epochs and the unsued samples
        train_dataset, n_samples_train, train_batch_size = tp['train_dataset'], tp['n_samples_train'], tp['train_batch_size']
        val_dataset, n_samples_val, val_batch_size = tp['validation_dataset'], tp['n_samples_val'], tp['val_batch_size']
        prediction_samples, step, closed_loop, connect = tp['prediction_samples'], tp['step'], tp['closed_loop'], tp['connect']
        tp['list_of_batch_indexes_train'], tp['train_step'], tp['list_of_batch_indexes_val'], tp['val_step'] = 0, 0, 0, 0
        if tp['prediction_samples'] >= 0:
            list_of_batch_indexes = \
                range(0, (n_samples_train - train_batch_size - prediction_samples + 1), (train_batch_size + step))
            check(n_samples_train - train_batch_size - prediction_samples + 1 > 0, ValueError,
                  f"The number of available sample are (n_samples_train ({n_samples_train}) - train_batch_size ({train_batch_size}) - prediction_samples ({prediction_samples}) + 1) = {n_samples_train - train_batch_size - prediction_samples + 1}.")
            tp['update_per_epochs'] = \
                (n_samples_train - train_batch_size - prediction_samples + 1) // (train_batch_size + step) + 1
            tp['unused_samples'] = n_samples_train - list_of_batch_indexes[-1] - train_batch_size - prediction_samples
            tp['list_of_batch_indexes_train'], tp['train_step'] = \
                self.__get_batch_indexes(tp, train_dataset, n_samples_train, prediction_samples, train_batch_size, step, type='train')
            if n_samples_val > 0:
                tp['list_of_batch_indexes_val'], tp['val_step'] = \
                    self.__get_batch_indexes(tp, val_dataset, n_samples_val, prediction_samples, val_batch_size, step, type='val')
        else:
            tp['update_per_epochs'] = (n_samples_train - train_batch_size) // train_batch_size + 1
            tp['unused_samples'] = n_samples_train - tp['update_per_epochs'] * train_batch_size
        return tp['list_of_batch_indexes_train'], tp['train_step'], tp['list_of_batch_indexes_val'], tp['val_step']

    def __training_info(self, tp, select_model, select_model_params, early_stopping, early_stopping_params, num_of_epochs, minimize_gain):
        ## Get early stopping
        early_stopping = self.__get_parameter(tp, early_stopping=early_stopping)
        if early_stopping:
            tp['early_stopping'] = early_stopping.__name__
        early_stopping_params = self.__get_parameter(tp, early_stopping_params=early_stopping_params)

        ## Get num_of_epochs
        num_of_epochs = self.__get_parameter(tp, num_of_epochs=num_of_epochs)

        ## Define the loss functions
        minimize_gain = self.__get_parameter(tp, minimize_gain=minimize_gain)
        tp['minimizers'] = {}
        for name, values in self._model_def['Minimizers'].items():
            self.__loss_functions[name] = CustomLoss(values['loss'])
            tp['minimizers'][name] = {}
            tp['minimizers'][name]['A'] = values['A']
            tp['minimizers'][name]['B'] = values['B']
            tp['minimizers'][name]['loss'] = values['loss']
            if name in minimize_gain:
                tp['minimizers'][name]['gain'] = minimize_gain[name]

        ## Select the model
        select_model = self.__get_parameter(tp, select_model=select_model)
        select_model_params = self.__get_parameter(tp, select_model_params=select_model_params)
        return minimize_gain, num_of_epochs

    def __check_needed_keys(self, tp):
        # Needed keys
        keys = set(self._model_def['Inputs'].keys())
        keys |= ({value['A'] for value in self._model_def['Minimizers'].values()} |
                 {value['B'] for value in  self._model_def['Minimizers'].values()})
        # Available keys
        keys -= set(self._model_def['Outputs'].keys()|self._model_def['Relations'].keys())
        keys -= set(self._model_def.recurrentInputs().keys())
        keys -= (set(tp['connect'].keys()|tp['closed_loop'].keys()))
        # Check if the keys are in the dataset
        check(set(keys).issubset(set(tp['XY_train'].keys())), KeyError,
              f"Not all the mandatory keys {keys} are present in the training dataset {set(tp['XY_train'].keys())}.")

    @enforce_types
    def trainModel(self,
                   models: str | list | None = None,
                   train_dataset: str | None = None, validation_dataset: str | None = None, test_dataset: str | None = None, splits: list | None = None,
                   closed_loop: dict | None = None, connect: dict | None = None, step: int | None = None, prediction_samples: int | None = None,
                   shuffle_data: bool | None = None,
                   early_stopping: Callable | None = None, early_stopping_params: dict | None = None,
                   select_model: Callable | None = None, select_model_params: dict | None = None,
                   minimize_gain: dict | None = None,
                   num_of_epochs: int = None,
                   train_batch_size: int = None, val_batch_size: int = None, test_batch_size: int = None,
                   optimizer: str | Optimizer | None = None,
                   lr: int | float | None = None, lr_param: dict | None = None,
                   optimizer_params: list | None = None, optimizer_defaults: dict | None = None,
                   training_params: dict | None = None,
                   add_optimizer_params: list | None = None, add_optimizer_defaults: dict | None = None
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
        ## Preliminary Checks
        tp = copy.deepcopy(self.__standard_train_parameters)
        self.__preliminary_checks()

        ## Get running parameter from dict
        self.__get_train_parameters(tp, training_params)

        # Setup recurrent train info
        prediction_samples, step, closed_loop, connect = \
            self.__setup_recurrent_train(tp, prediction_samples, step, closed_loop, connect)

        ## Get the dataset
        shuffle_data, XY_train, XY_val, XY_test, n_samples_train, n_samples_val, n_samples_test = \
            self.__setup_dataset(tp, shuffle_data, train_dataset, validation_dataset, test_dataset, splits)

        ## Get batchsize
        train_batch_size, val_batch_size, test_batch_size = \
            self.__get_batch_sizes(tp, train_batch_size, val_batch_size, test_batch_size)

        ## Define batch indexes
        list_of_batch_indexes_train, train_step, list_of_batch_indexes_val, val_step = \
            self.__setup_batch_indexes(tp)

        ## Define the optimizer
        self.__optimizer = \
            self.__inizilize_optimizer(tp, optimizer, optimizer_params, optimizer_defaults, add_optimizer_params,
                                               add_optimizer_defaults, models, lr, lr_param)

        ## Define mandatory inputs
        mandatory_inputs, non_mandatory_inputs = self._get_mandatory_inputs(connect, closed_loop)

        ## Get the training parameters
        minimize_gain, num_of_epochs = \
            self.__training_info(tp, select_model, select_model_params, early_stopping, early_stopping_params, num_of_epochs, minimize_gain)

        ## Check the needed keys are in the datasets
        self.__check_needed_keys(tp)

        ## Check close loop and connect
        self._clean_log_internal()

        self.run_training_params = tp
        ## Clean the dict of the training parameter
        #del self.run_training_params['minimize_gain']
        del self.run_training_params['lr']
        del self.run_training_params['lr_param']
        #if prediction_samples < 0:
            #del self.run_training_params['connect']
            #del self.run_training_params['closed_loop']
            #del self.run_training_params['step']
        if early_stopping is None:
            del self.run_training_params['early_stopping']
            del self.run_training_params['early_stopping_params']

        ## Create the train, validation and test loss dictionaries
        train_losses, val_losses = {}, {}
        for key in self._model_def['Minimizers'].keys():
            train_losses[key] = []
            if n_samples_val > 0:
                val_losses[key] = []

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

        selected_model_def = ModelDef(self._model_def.getJson())

        ## Show the training parameters
        self.visualizer.showTrainParams()

        import time
        ## start the train timer
        start = time.time()
        self.visualizer.showStartTraining()

        ## Update with virtual states
        #TODO Added the case of prediction_samples = -1 removed the connect
        self._model.update(closed_loop=closed_loop, connect=connect)

        self.resetStates()  ## Reset the states

        for epoch in range(num_of_epochs):
            ## TRAIN
            self._model.train()
            if prediction_samples >= 0:
                losses = self._recurrent_inference(XY_train, list_of_batch_indexes_train, train_batch_size, minimize_gain,
                                                   prediction_samples, train_step, non_mandatory_inputs,
                                                   mandatory_inputs, self.__loss_functions,
                                                   shuffle=shuffle_data, optimizer=self.__optimizer)
            else:
                losses = self._inference(XY_train, n_samples_train, train_batch_size, minimize_gain,
                                         self.__loss_functions, shuffle=shuffle_data, optimizer=self.__optimizer)

            ## save the losses
            for ind, key in enumerate(self._model_def['Minimizers'].keys()):
                train_losses[key].append(torch.mean(losses[ind]).tolist())

            if n_samples_val > 0:
                ## VALIDATION
                self._model.eval()
                setted_log_internal = self._log_internal
                self._set_log_internal(False)  # TODO To remove when the function is moved outside the train
                if prediction_samples >= 0:
                    losses = self._recurrent_inference(XY_val, list_of_batch_indexes_val, val_batch_size, minimize_gain,
                                                       prediction_samples, val_step, non_mandatory_inputs,
                                                       mandatory_inputs, self.__loss_functions)
                else:
                    losses = self._inference(XY_val, n_samples_val, val_batch_size, minimize_gain,
                                             self.__loss_functions)

                self._set_log_internal(setted_log_internal)

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

        ## Remove virtual states
        self._removeVirtualStates(connect, closed_loop)

        ## Get trained model from torch and set the model_def
        self._model_def.updateParameters(self._model)

        return tp
#840