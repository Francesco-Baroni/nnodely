import  torch

from nnodely.support.utils import  TORCH_DTYPE, check, enforce_types
from nnodely.basic.modeldef import ModelDef

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

