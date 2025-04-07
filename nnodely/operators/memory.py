import  torch

from nnodely.utils import  TORCH_DTYPE, check

class Memory:
    def __init__(self):
        check(type(self) is not Memory, TypeError, "Loader class cannot be instantiated directly")

        # Model definition
        self.states = {}
        self.input_n_samples = {}
        self.max_n_samples = 0

    def resetStates(self, states=[], batch=1):
        if states: ## reset only specific states
            for key in states:
                window_size = self.input_n_samples[key]
                dim = self.model_def['States'][key]['dim']
                self.states[key] = torch.zeros(size=(batch, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)
        else: ## reset all states
            self.states = {}
            for key, state in self.model_def['States'].items():
                window_size = self.input_n_samples[key]
                dim = state['dim']
                self.states[key] = torch.zeros(size=(batch, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)

    def _updateState(self, X, out_closed_loop, out_connect):
        ## Update
        for key, val in out_closed_loop.items():
            shift = val.shape[1]  ## take the output time dimension
            X[key] = torch.roll(X[key], shifts=-1, dims=1)  ## Roll the time window
            X[key][:, -shift:, :] = val  ## substitute with the predicted value
            self.states[key] = X[key].clone().detach()
        for key, value in out_connect.items():
            X[key] = value
            self.states[key] = X[key].clone().detach()
