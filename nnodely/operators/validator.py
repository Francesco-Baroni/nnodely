import torch

import numpy as np

from nnodely.support.utils import ReadOnlyDict

from nnodely.basic.loss import CustomLoss
from nnodely.operators.network import Network
from nnodely.support.utils import  check, TORCH_DTYPE, enforce_types

class Validator(Network):
    @enforce_types
    def __init__(self):
        check(type(self) is not Validator, TypeError, "Validator class cannot be instantiated directly")
        super().__init__()

        # Validation Parameters
        self.__performance = {}
        self.__prediction = {}

    @property
    def performance(self):
        return ReadOnlyDict(self.__performance)

    @property
    def prediction(self):
        return ReadOnlyDict(self.__prediction)
    
    @enforce_types
    def resultAnalysis(self,
                       dataset: str | None = None,
                       XY: dict | None = None,
                       minimize_gain: dict = {},
                       closed_loop: dict = {},
                       connect: dict = {},
                       prediction_samples: int | str | None = None,
                       step: int = 0,
                       batch_size: int | None = None
                       ) -> None:

        import warnings
        with torch.enable_grad() if self._caluclate_grad() else torch.inference_mode():
            ## Init model for retults analysis
            self._model.eval()
            self.__performance[dataset] = {}
            self.__prediction[dataset] = {}
            A = {}
            B = {}
            total_losses = {}

            # Create the losses
            losses = {}
            for name, values in self._model_def['Minimizers'].items():
                losses[name] = CustomLoss(values['loss'])

            recurrent = False
            if (closed_loop or connect or len(self._model_def.recurrentInputs()) > 0) and prediction_samples is not None:
                recurrent = True

            if XY is None:
                check(dataset in self._data.keys(), ValueError, f'The dataset {dataset} is not loaded!')
                XY = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in self._data[dataset].items()}
            #TODO check that the dataset name is not already laaded if I pass XY
            n_samples = len(XY[list(XY.keys())[0]])

            if recurrent:
                batch_size = batch_size if batch_size is not None else n_samples - prediction_samples

                model_inputs = list(self._model_def['Inputs'].keys())

                state_closed_loop = [key for key, value in self._model_def['Inputs'].items() if 'closedLoop' in value.keys()] + list(closed_loop.keys())
                state_connect = [key for key, value in self._model_def['Inputs'].items() if 'connect' in value.keys()] + list(connect.keys())

                non_mandatory_inputs = state_closed_loop + state_connect
                mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))

                for key, value in self._model_def['Minimizers'].items():
                    total_losses[key], A[key], B[key] = [], [], []
                    for horizon_idx in range(prediction_samples + 1):
                        A[key].append([])
                        B[key].append([])

                #TODO this is not correct the data are not depending on train val test
                list_of_batch_indexes = list(range(n_samples - prediction_samples))
                ## Remove forbidden indexes in case of a multi-file dataset
                if dataset in self._multifile.keys(): ## Multi-file Dataset
                    if n_samples == self.run_training_params['n_samples_train']: ## Training
                        list_of_batch_indexes, step = self.__get_batch_indexes(dataset, n_samples, prediction_samples, batch_size, step, type='train')
                    elif n_samples == self.run_training_params['n_samples_val']: ## Validation
                        list_of_batch_indexes, step = self.__get_batch_indexes(dataset, n_samples, prediction_samples, batch_size, step, type='val')
                    else:
                        list_of_batch_indexes, step = self.__get_batch_indexes(dataset, n_samples, prediction_samples, batch_size, step, type='test')

                X = {}
                ## Update with virtual states
                self._model.update(closed_loop=closed_loop, connect=connect)
                while len(list_of_batch_indexes) >= batch_size:
                    idxs = list_of_batch_indexes[:batch_size]
                    for num in idxs:
                        list_of_batch_indexes.remove(num)
                    if step > 0:
                        if len(list_of_batch_indexes) >= step:
                            step_idxs =  list_of_batch_indexes[:step]
                            for num in step_idxs:
                                list_of_batch_indexes.remove(num)
                        else:
                            list_of_batch_indexes = []
                    ## Reset
                    horizon_losses = {key: [] for key in self._model_def['Minimizers'].keys()}
                    for key in non_mandatory_inputs:
                        if key in XY.keys():
                            ## with data
                            X[key] = XY[key][idxs]
                        else:  ## with zeros
                            window_size = self._input_n_samples[key]
                            dim = self.json['Inputs'][key]['dim']
                            if 'type' in self.json['Inputs'][key]:
                                X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=TORCH_DTYPE, requires_grad=True)
                            else:
                                X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)
                            self._states[key] = X[key]

                    for horizon_idx in range(prediction_samples + 1):
                        ## Get data
                        for key in mandatory_inputs:
                            X[key] = XY[key][[idx+horizon_idx for idx in idxs]]
                        ## Forward pass
                        out, minimize_out, out_closed_loop, out_connect = self._model(X)

                        ## Loss Calculation
                        for key, value in self._model_def['Minimizers'].items():
                            A[key][horizon_idx].append(minimize_out[value['A']].detach().numpy())
                            B[key][horizon_idx].append(minimize_out[value['B']].detach().numpy())
                            loss = losses[key](minimize_out[value['A']], minimize_out[value['B']])
                            loss = (loss * minimize_gain[key]) if key in minimize_gain.keys() else loss  ## Multiply by the gain if necessary
                            horizon_losses[key].append(loss)

                        ## Update
                        self._updateState(X, out_closed_loop, out_connect)

                    ## Calculate the total loss
                    for key in self._model_def['Minimizers'].keys():
                        loss = sum(horizon_losses[key]) / (prediction_samples + 1)
                        total_losses[key].append(loss.detach().numpy())

                for key, value in self._model_def['Minimizers'].items():
                    for horizon_idx in range(prediction_samples + 1):
                        A[key][horizon_idx] = np.concatenate(A[key][horizon_idx])
                        B[key][horizon_idx] = np.concatenate(B[key][horizon_idx])
                    total_losses[key] = np.mean(total_losses[key])

            else:
                if batch_size is None:
                    batch_size = n_samples

                for key, value in self._model_def['Minimizers'].items():
                    total_losses[key], A[key], B[key] = [], [], []

                for idx in range(0, (n_samples - batch_size + 1), batch_size):
                    ## Build the input tensor
                    X = {key: val[idx:idx + batch_size] for key, val in XY.items()}

                    ## Model Forward
                    _, minimize_out, _, _ = self._model(X)  ## Forward pass
                    ## Loss Calculation
                    for key, value in self._model_def['Minimizers'].items():
                        A[key].append(minimize_out[value['A']].detach().numpy())
                        B[key].append(minimize_out[value['B']].detach().numpy())
                        loss = losses[key](minimize_out[value['A']], minimize_out[value['B']])
                        loss = (loss * minimize_gain[key]) if key in minimize_gain.keys() else loss
                        total_losses[key].append(loss.detach().numpy())

                for key, value in self._model_def['Minimizers'].items():
                    A[key] = np.concatenate(A[key])
                    B[key] = np.concatenate(B[key])
                    total_losses[key] = np.mean(total_losses[key])

            for ind, (key, value) in enumerate(self._model_def['Minimizers'].items()):
                A_np = np.array(A[key])
                B_np = np.array(B[key])
                self.__performance[dataset][key] = {}
                self.__performance[dataset][key][value['loss']] = np.mean(total_losses[key]).item()
                self.__performance[dataset][key]['fvu'] = {}
                # Compute FVU
                residual = A_np - B_np
                error_var = np.var(residual)
                error_mean = np.mean(residual)
                #error_var_manual = np.sum((residual-error_mean) ** 2) / (len(self.__prediction['B'][ind]) - 0)
                #print(f"{key} var np:{new_error_var} and var manual:{error_var_manual}")
                with warnings.catch_warnings(record=True) as w:
                    self.__performance[dataset][key]['fvu']['A'] = (error_var / np.var(A_np)).item()
                    self.__performance[dataset][key]['fvu']['B'] = (error_var / np.var(B_np)).item()
                    if w and np.var(A_np) == 0.0 and  np.var(B_np) == 0.0:
                        self.__performance[dataset][key]['fvu']['A'] = np.nan
                        self.__performance[dataset][key]['fvu']['B'] = np.nan
                self.__performance[dataset][key]['fvu']['total'] = np.mean([self.__performance[dataset][key]['fvu']['A'],self.__performance[dataset][key]['fvu']['B']]).item()
                # Compute AIC
                #normal_dist = norm(0, error_var ** 0.5)
                #probability_of_residual = normal_dist.pdf(residual)
                #log_likelihood_first = sum(np.log(probability_of_residual))
                p1 = -len(residual)/2.0*np.log(2*np.pi)
                with warnings.catch_warnings(record=True) as w:
                    p2 = -len(residual)/2.0*np.log(error_var)
                    p3 = -1 / (2.0 * error_var) * np.sum(residual ** 2)
                    if w and p2 == np.float32(np.inf) and p3 == np.float32(-np.inf):
                        p2 = p3 = 0.0
                log_likelihood = p1+p2+p3
                #print(f"{key} log likelihood second mode:{log_likelihood} = {p1}+{p2}+{p3} first mode: {log_likelihood_first}")
                total_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
                #print(f"{key} total_params:{total_params}")
                aic = - 2 * log_likelihood + 2 * total_params
                #print(f"{key} aic:{aic}")
                self.__performance[dataset][key]['aic'] = {'value':aic,'total_params':total_params,'log_likelihood':log_likelihood}
                # Prediction and target
                self.__prediction[dataset][key] = {}
                self.__prediction[dataset][key]['A'] = A_np.tolist()
                self.__prediction[dataset][key]['B'] = B_np.tolist()

            self.__performance[dataset]['total'] = {}
            self.__performance[dataset]['total']['mean_error'] = np.mean([value for key,value in total_losses.items()])
            self.__performance[dataset]['total']['fvu'] = np.mean([self.__performance[dataset][key]['fvu']['total'] for key in self._model_def['Minimizers'].keys()])
            self.__performance[dataset]['total']['aic'] = np.mean([self.__performance[dataset][key]['aic']['value']for key in self._model_def['Minimizers'].keys()])

        self.visualizer.showResult(dataset)
