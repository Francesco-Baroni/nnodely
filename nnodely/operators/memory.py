import  torch

from nnodely.support.utils import  TORCH_DTYPE, check, enforce_types

class Memory:
    def __init__(self):
        check(type(self) is not Memory, TypeError, "Loader class cannot be instantiated directly")

        # Model definition
        self._states = {}
        self._input_n_samples = {}
        self._input_ns_backward = {}
        self._input_ns_forward = {}
        self._max_samples_backward = None
        self._max_samples_forward = None
        self._max_n_samples = 0

    def _removeVirtualStates(self, connect, closed_loop):
        for key in (connect.keys() | closed_loop.keys()):
            if key in self._states.keys():
                del self._states[key]

    def _updateState(self, X, out_closed_loop, out_connect):
        ## Update
        for key, val in out_closed_loop.items():
            shift = val.shape[1]  ## take the output time dimension
            X[key] = torch.roll(X[key], shifts=-1, dims=1)  ## Roll the time window
            X[key][:, -shift:, :] = val  ## substitute with the predicted value
            self._states[key] = X[key].clone().detach()
        for key, value in out_connect.items():
            X[key] = value
            self._states[key] = X[key].clone().detach()

    @enforce_types
    def resetStates(self, states:set={}, batch:int=1) -> None:
        if states: ## reset only specific states
            for key in states:
                window_size = self._input_n_samples[key]
                dim = self._model_def['States'][key]['dim']
                self._states[key] = torch.zeros(size=(batch, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)
        else: ## reset all states
            self._states = {}
            for key, state in self._model_def['States'].items():
                window_size = self._input_n_samples[key]
                dim = state['dim']
                self._states[key] = torch.zeros(size=(batch, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)

