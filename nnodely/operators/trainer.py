import copy, torch, time, inspect

from collections.abc import Callable
from functools import wraps

from nnodely.basic.modeldef import ModelDef
from nnodely.basic.model import Model
from nnodely.basic.optimizer import Optimizer, SGD, Adam
from nnodely.basic.loss import CustomLoss
from nnodely.operators.network import Network
from nnodely.support.utils import check, enforce_types
from nnodely.basic.relation import Stream
from nnodely.layers.output import Output

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

class Trainer(Network):
    def __init__(self):
        check(type(self) is not Trainer, TypeError, "Trainer class cannot be instantiated directly")
        super().__init__()

        # Default Training Parameters
        self.__standard_train_parameters = {
            'models' : None,
            'train_dataset' : None, 'validation_dataset' : None, 
            'dataset' : None, 'splits' : [100, 0, 0],
            'closed_loop' : {}, 'connect' : {}, 'step' : 0, 'prediction_samples' : 0,
            'shuffle_data' : True,
            'early_stopping' : None, 'early_stopping_params' : {},
            'select_model' : 'last', 'select_model_params' : {},
            'minimize_gain' : {},
            'num_of_epochs': 100,
            'train_batch_size' : 128, 'val_batch_size' : 128,
            'optimizer' : 'Adam',
            'lr' : 0.001, 'lr_param' : {},
            'optimizer_params' : [], 'add_optimizer_params' : [],
            'optimizer_defaults' : {}, 'add_optimizer_defaults' : {}
        }
        ## User Parameters
        self.running_parameters = {}

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
        check(self._data_loaded, RuntimeError, 'There is no data loaded! The Training will stop.')
        check('Models' in self._model_def.getJson(), RuntimeError, 'There are no models to train. Load a model using the addModel function.')
        check(list(self._model.parameters()), RuntimeError, 'There are no modules with learnable parameters! The Training will stop.')

    def fill_parameters(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            # Get standard parameters
            standard = self.__standard_train_parameters
            # Get user_parameters
            users = bound.arguments.get('training_params', None)
            # Fill missing (None) arguments
            for param in sig.parameters.values():
                if param.name == 'self':
                    continue
                if bound.arguments.get(param.name, None) is None:
                    if param.name in users.keys():
                        bound.arguments[param.name] = users[param.name]
                    else:
                        bound.arguments[param.name] = standard.get(param.name, None)
            return func(**bound.arguments)
        return wrapper
    
    def __clip_batch_size(self, n_samples, batch_size=None, prediction_samples=None):
        prediction = 0 if prediction_samples == -1 else prediction_samples #This value is used to disconnect the connect
        batch_size = batch_size if batch_size <= n_samples - prediction else max(0, n_samples - prediction)
        check((n_samples - batch_size + 1) > 0, ValueError, f"The number of available sample are {n_samples - batch_size + 1}")
        check(batch_size > 0, ValueError, f'The batch_size must be greater than 0.')
        return batch_size
    
    def __clip_step(self, step, batch_indexes, batch_size):
        clipped_step = copy.deepcopy(step)
        if clipped_step < 0:  ## clip the step to zero
            log.warning(f"The step is negative ({clipped_step}). The step is set to zero.", stacklevel=5)
            clipped_step = 0
        if clipped_step > (len(batch_indexes) - batch_size):  ## Clip the step to the maximum number of samples
            log.warning(f"The step ({clipped_step}) is greater than the number of available samples ({len(batch_indexes) - batch_size}). The step is set to the maximum number.", stacklevel=5)
            clipped_step = len(batch_indexes) - batch_size
        check((batch_size + clipped_step) > 0, ValueError, f"The sum of batch_size={batch_size} and the step={clipped_step} must be greater than 0.")
        return clipped_step

    def __initialize_optimizer(self, models, optimizer, optimizer_params, optimizer_defaults, add_optimizer_defaults, add_optimizer_params, lr, lr_param):
        ## Get models
        json_models = []
        if 'Models' in self._model_def:
            json_models = list(self._model_def['Models'].keys()) if type(self._model_def['Models']) is dict else [self._model_def['Models']]
        if models is None:
            models = json_models
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
            check(issubclass(type(optimizer), Optimizer), TypeError, "The optimizer must be an Optimizer or str")

        optimizer.set_params_to_train(self._model.all_parameters, params_to_train)
        optimizer.add_defaults('lr', lr)
        optimizer.add_option_to_params('lr', lr_param)
        if optimizer_defaults != {}:
            optimizer.set_defaults(optimizer_defaults)
        if optimizer_params != []:
            optimizer.set_params(optimizer_params)
        for key, value in add_optimizer_defaults.items():
            optimizer.add_defaults(key, value)
        add_optimizer_params = optimizer.unfold(add_optimizer_params)
        for param in add_optimizer_params:
            par = param['params']
            for key, value in param.items():
                optimizer.add_option_to_params(key, {par: value})
        # Modify the parameter
        optimizer.add_defaults('lr', lr)
        if 'lr' in optimizer_defaults:
            optimizer.add_defaults('lr', optimizer_defaults['lr']) 
        if 'lr' in lr_param:
            optimizer.add_option_to_params('lr', lr_param)
        self.__optimizer = optimizer.get_torch_optimizer()
    
    def __initialize_loss(self):
        for name, values in self._model_def['Minimizers'].items():
            self.__loss_functions[name] = CustomLoss(values['loss'])

    def get_training_info(self):
        tp = copy.deepcopy(self.running_parameters)

        ## Dataset
        if tp['train_dataset'] is None and tp['dataset'] is None:
            tp['dataset'] = list(self._data.keys())[0]

        ## training
        tp['update_per_epochs'] = len(tp['train_indexes']) // (tp['train_batch_size'] + tp['step']) 
        total_samples = len(tp['train_indexes']) + tp['prediction_samples']  ## number of samples taking into account the prediction horizon
        tp['unused_samples'] = (total_samples - tp['update_per_epochs'] * tp['train_batch_size']) - tp['prediction_samples']  ## number of samples not used for training

        ## optimizer
        tp['optimizer_defaults'] = self.__optimizer.defaults

        ## early stopping
        early_stopping = tp['early_stopping']
        if early_stopping:
            tp['early_stopping'] = early_stopping.__name__

        ## Loss functions
        tp['minimizers'] = {}
        for name, values in self._model_def['Minimizers'].items():
            tp['minimizers'][name] = {}
            tp['minimizers'][name]['A'] = values['A']
            tp['minimizers'][name]['B'] = values['B']
            tp['minimizers'][name]['loss'] = values['loss']
            if name in tp['minimize_gain']:
                tp['minimizers'][name]['gain'] = tp['minimize_gain'][name]

        ## Remove useless information
        del tp['train_indexes']
        del tp['val_indexes']
        del tp['XY_train']
        del tp['XY_val']
        return tp
        
    def __check_needed_keys(self, train_data, connect, closed_loop):
        # Needed keys
        keys = set(self._model_def['Inputs'].keys())
        keys |= ({value['A'] for value in self._model_def['Minimizers'].values()} | {value['B'] for value in  self._model_def['Minimizers'].values()})
        # Available keys
        keys -= set(self._model_def['Outputs'].keys()|self._model_def['Relations'].keys())
        keys -= set(self._model_def.recurrentInputs().keys())
        keys -= (set(connect.keys()|closed_loop.keys()))
        # Check if the keys are in the dataset
        check(set(keys).issubset(set(train_data.keys())), KeyError, f"Not all the mandatory keys {keys} are present in the training dataset {set(train_data.keys())}.")

    @enforce_types
    @fill_parameters
    def trainModel(self, *,
                   models: str | list | None = None,
                   train_dataset: str | list | None = None, validation_dataset: str | list | None = None, 
                   dataset: str | list | None = None, splits: list | None = None,
                   closed_loop: dict | None = None, connect: dict | None = None, step: int | None = None, prediction_samples: int | None = None,
                   shuffle_data: bool | None = None,
                   early_stopping: Callable | None = None, early_stopping_params: dict | None = None,
                   select_model: Callable | None = None, select_model_params: dict | None = None,
                   minimize_gain: dict | None = None,
                   num_of_epochs: int = None,
                   train_batch_size: int = None, val_batch_size: int = None,
                   optimizer: str | Optimizer | None = None,
                   lr: int | float | None = None, lr_param: dict | None = None,
                   optimizer_params: list | None = None, optimizer_defaults: dict | None = None,
                   add_optimizer_params: list | None = None, add_optimizer_defaults: dict | None = None,
                   training_params: dict | None = {}
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
        self.__preliminary_checks()

        ## Recurret variables
        prediction_samples = self._setup_recurrent_variables(prediction_samples, closed_loop, connect)

        ## Get the dataset
        XY_train, XY_val, _, n_samples_train, n_samples_val, n_samples_test, train_indexes, val_indexes, _ = self._setup_dataset(train_dataset, validation_dataset, None, dataset, splits, prediction_samples)
        self.__check_needed_keys(train_data=XY_train, connect=connect, closed_loop=closed_loop)

        ## Get batch size
        train_batch_size = self.__clip_batch_size(n_samples_train, train_batch_size, prediction_samples)
        if n_samples_val > 0:
            val_batch_size = self.__clip_batch_size(n_samples_val, val_batch_size, prediction_samples)

        ## Clip the step
        train_step = self.__clip_step(step, train_indexes, train_batch_size)
        if n_samples_val > 0:
            val_step = self.__clip_step(step, val_indexes, val_batch_size)
        ## Save the training parameters
        self.running_parameters = copy.deepcopy({key:value for key,value in locals().items() if key not in ['self', 'kwargs', 'training_params']})

        ## Define the optimizer
        self.__initialize_optimizer(models, optimizer, optimizer_params, optimizer_defaults, add_optimizer_defaults, add_optimizer_params, lr, lr_param)

        ## Define the loss functions
        self.__initialize_loss()

        ## Define mandatory inputs
        mandatory_inputs, non_mandatory_inputs = self._get_mandatory_inputs(connect, closed_loop)

        ## Check close loop and connect
        self._clean_log_internal()

        ## Create the train, validation and test loss dictionaries
        train_losses, val_losses = {}, {}
        for key in self._model_def['Minimizers'].keys():
            train_losses[key] = []
            if n_samples_val > 0:
                val_losses[key] = []

        ## Set the gradient to true if necessary
        model_inputs = self._model_def['Inputs']
        for key in model_inputs.keys():
            if 'type' in model_inputs[key]:
                if key in XY_train:
                    XY_train[key].requires_grad_(True)
                if key in XY_val:
                    XY_val[key].requires_grad_(True)
        selected_model_def = ModelDef(self._model_def.getJson())

        ## Show the training parameters
        self.visualizer.showTrainParams()
        self.visualizer.showStartTraining()

        ## Update with virtual states
        self._model.update(closed_loop=closed_loop, connect=connect)
        self.resetStates()  ## Reset the states

        ## start the train timer
        start = time.time()
        for epoch in range(num_of_epochs):
            ## TRAIN
            self._model.train()
            if prediction_samples >= 0:
                losses = self._recurrent_inference(XY_train, train_indexes, train_batch_size, minimize_gain, prediction_samples, train_step,
                                                    non_mandatory_inputs, mandatory_inputs, self.__loss_functions, shuffle=shuffle_data, optimizer=self.__optimizer)
            else:
                losses = self._inference(XY_train, n_samples_train, train_batch_size, minimize_gain, self.__loss_functions, shuffle=shuffle_data, optimizer=self.__optimizer)
            ## save the losses
            for ind, key in enumerate(self._model_def['Minimizers'].keys()):
                train_losses[key].append(torch.mean(losses[ind]).tolist())

            if n_samples_val > 0:
                ## VALIDATION
                self._model.eval()
                setted_log_internal = self._log_internal
                self._set_log_internal(False)  # TODO To remove when the function is moved outside the train
                if prediction_samples >= 0:
                    losses = self._recurrent_inference(XY_val, val_indexes, val_batch_size, minimize_gain, prediction_samples, val_step, 
                                                       non_mandatory_inputs, mandatory_inputs, self.__loss_functions)
                else:
                    losses = self._inference(XY_val, n_samples_val, val_batch_size, minimize_gain, self.__loss_functions)
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

        ## Visualize the training time
        end = time.time()
        self.visualizer.showTrainingTime(end - start)

        for key in self._model_def['Minimizers'].keys():
            self._training[key] = {'train': train_losses[key]}
            if n_samples_val > 0:
                self._training[key]['val'] = val_losses[key]
        self.visualizer.showEndTraining(num_of_epochs - 1, train_losses, val_losses)

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
        return self.get_training_info()
#from 685
#from 840