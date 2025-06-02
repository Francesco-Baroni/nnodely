import torch, warnings

import numpy as np

from nnodely.support.utils import ReadOnlyDict, get_batch_size

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
                       data: dict | None = None,
                       minimize_gain: dict = {},
                       closed_loop: dict = {},
                       connect: dict = {},
                       prediction_samples: int | str = -1, #TODO uniform to training set to 0
                       step: int = 0,
                       batch_size: int | None = None
                       ) -> None:

        with torch.enable_grad() if self._get_gradient_on_inference() else torch.inference_mode():
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

            if data is None:
                check(dataset in self._data.keys(), ValueError, f'The dataset {dataset} is not loaded!')
                data = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in self._data[dataset].items()}
            n_samples = len(data[list(data.keys())[0]])

            batch_size = get_batch_size(n_samples, batch_size, prediction_samples)
            prediction_samples = self._setup_recurrent_variables(prediction_samples, closed_loop, connect)

            if prediction_samples >= 0:
                mandatory_inputs, non_mandatory_inputs = self._get_mandatory_inputs(connect,closed_loop)

                for key, value in self._model_def['Minimizers'].items():
                    total_losses[key], A[key], B[key] = [], [], []
                    for horizon_idx in range(prediction_samples + 1):
                        A[key].append([])
                        B[key].append([])

                # #TODO FIXME
                # list_of_batch_indexes = list(range(n_samples - prediction_samples))
                # ## Remove forbidden indexes in case of a multi-file dataset
                # if dataset in self._multifile.keys(): ## Multi-file Dataset
                #     if n_samples == self.run_training_params['n_samples_train']: ## Training
                #         list_of_batch_indexes, step = self.__get_batch_indexes(dataset, n_samples, prediction_samples, batch_size, step, type='train')
                #     elif n_samples == self.run_training_params['n_samples_val']: ## Validation
                #         list_of_batch_indexes, step = self.__get_batch_indexes(dataset, n_samples, prediction_samples, batch_size, step, type='val')
                #     else:
                #         list_of_batch_indexes, step = self.__get_batch_indexes(dataset, n_samples, prediction_samples, batch_size, step, type='test')
                # #FIXME
                if dataset and dataset in self._multifile.keys(): ## Multi-file Dataset
                    batch_indexes = self._get_batch_indexes(dataset, prediction_samples)
                else:
                    batch_indexes = list(range(n_samples - prediction_samples))

                ## Update with virtual states
                self._model.update(closed_loop=closed_loop, connect=connect)
                self._recurrent_inference(data, batch_indexes, batch_size, minimize_gain, prediction_samples,
                                          step, non_mandatory_inputs, mandatory_inputs, losses,
                                          total_losses = total_losses, A = A, B = B)

                for key, value in self._model_def['Minimizers'].items():
                    for horizon_idx in range(prediction_samples + 1):
                        if A is not None:
                            A[key][horizon_idx] = np.concatenate(A[key][horizon_idx])
                        if B is not None:
                            B[key][horizon_idx] = np.concatenate(B[key][horizon_idx])
                    if total_losses is not None:
                        total_losses[key] = np.mean(total_losses[key])
            else:
                for key, value in self._model_def['Minimizers'].items():
                    total_losses[key], A[key], B[key] = [], [], []

                self._inference(data, n_samples, batch_size, minimize_gain, losses, total_losses = total_losses, A = A, B = B)

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
#227