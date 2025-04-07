import os, random

import pandas as pd
import numpy as np
import pandas.api.types as ptypes

from nnodely.support.utils import check, log

class Loader:
    def __init__(self):
        check(type(self) is not Loader, TypeError, "Loader class cannot be instantiated directly")

        # Dataaset Parameters
        self.__n_datasets = 0
        self.__datasets_loaded = set()

        self._data_loaded = False
        self._file_count = 0
        self._num_of_samples = {}
        self._data = {}
        self._multifile = {}

    def getSamples(self, dataset, index = None, window=1):
        """
        Retrieves a window of samples from a given dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to retrieve samples from.
        index : int, optional
            The starting index of the samples. If None, a random index is chosen. Default is None.
        window : int, optional
            The number of consecutive samples to retrieve. Default is 1.

        Returns
        -------
        dict
            A dictionary containing the retrieved samples. The keys are input and state names, and the values are lists of samples.

        Raises
        ------
        ValueError
            If the dataset is not loaded.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/dataset.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.loadData('dataset_name')
            >>> samples = model.getSamples('dataset_name', index=10, window=5)
        """
        if index is None:
            index = random.randint(0, self._num_of_samples[dataset] - window)
        check(self._data_loaded, ValueError, 'The Dataset must first be loaded using <loadData> function!')
        if self._data_loaded:
            result_dict = {}
            for key in (self.model_def['Inputs'].keys() | self.model_def['States'].keys()):
                result_dict[key] = []
            for idx in range(window):
                for key ,samples in self._data[dataset].items():
                    if key in (self.model_def['Inputs'].keys() | self.model_def['States'].keys()):
                        result_dict[key].append(samples[index+idx])
            return result_dict

    def filterData(self, filter_function, dataset_name = None):
        """
        Filters the data in the dataset using the provided filter function.

        Parameters
        ----------
        filter_function : Callable
            A function that takes a sample as input and returns True if the sample should be kept, and False if it should be removed.
        dataset_name : str or None, optional
            The name of the dataset to filter. If None, all datasets are filtered. Default is None.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/dataset.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.loadData('dataset_name', 'path/to/data')
            >>> def filter_fn(sample):
            >>>     return sample['input1'] > 0
            >>> model.filterData(filter_fn, 'dataset_name')
        """
        idx_to_remove = []
        if dataset_name is None:
            for name in self._data.keys():
                dataset = self._data[name]
                n_samples = len(dataset[list(dataset.keys())[0]])

                data_for_filter = []
                for i in range(n_samples):
                    new_sample = {key: val[i] for key, val in dataset.items()}
                    data_for_filter.append(new_sample)

                for idx, sample in enumerate(data_for_filter):
                    if not filter_function(sample):
                        idx_to_remove.append(idx)

                for key in self._data[name].keys():
                    self._data[name][key] = np.delete(self._data[name][key], idx_to_remove, axis=0)
                    self._num_of_samples[name] = self._data[name][key].shape[0]
                self.visualizer.showDataset(name=name)

        else:
            dataset = self._data[dataset_name]
            n_samples = len(dataset[list(dataset.keys())[0]])

            data_for_filter = []
            for i in range(n_samples):
                new_sample = {key: val[i] for key, val in dataset.items()}
                data_for_filter.append(new_sample)

            for idx, sample in enumerate(data_for_filter):
                if not filter_function(sample):
                    idx_to_remove.append(idx)

            for key in self._data[dataset_name].keys():
                self._data[dataset_name][key] = np.delete(self._data[dataset_name][key], idx_to_remove, axis=0)
                self._num_of_samples[dataset_name] = self._data[dataset_name][key].shape[0]
            self.visualizer.showDataset(name=dataset_name)

    def loadData(self, name, source, format=None, skiplines=0, delimiter=',', header=None, resampling=False):
        """
        Loads data into the model. The data can be loaded from a directory path containing the csv files or from a crafted dataset.

        Parameters
        ----------
        name : str
            The name of the dataset.
        source : str or list
            The source of the data. Can be a directory path containing the csv files or a list of custom data.
        format : list or None, optional
            The format of the data. When loading multiple csv files the format parameter will define how to read each column of the file. Default is None.
        skiplines : int, optional
            The number of lines to skip at the beginning of the file. Default is 0.
        delimiter : str, optional
            The delimiter used in the data files. Default is ','.
        header : list or None, optional
            The header of the data files. Default is None.

        Raises
        ------
        ValueError
            If the network is not neuralized.
            If the delimiter is not valid.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/dataset.ipynb
            :alt: Open in Colab

        Example - load data from files:
            >>> x = Input('x')
            >>> y = Input('y')
            >>> out = Output('out',Fir(x.tw(0.05)))
            >>> test = Modely(visualizer=None)
            >>> test.addModel('example_model', out)
            >>> test.neuralizeModel(0.01)
            >>> data_struct = ['x', '', 'y']
            >>> test.loadData(name='example_dataset', source='path/to/data', format=data_struct)

        Example - load data from a crafted dataset:
            >>> x = Input('x')
            >>> y = Input('y')
            >>> out = Output('out',Fir(x.tw(0.05)))
            >>> test = Modely(visualizer=None)
            >>> test.addModel('example_model', out)
            >>> test.neuralizeModel(0.01)
            >>> data_x = np.array(range(10))
            >>> dataset = {'x': data_x, 'y': (2*data_x)}
            >>> test.loadData(name='example_dataset',source=dataset)
        """
        check(self.neuralized, ValueError, "The network is not neuralized.")
        check(delimiter in ['\t', '\n', ';', ',', ' '], ValueError, 'delimiter not valid!')

        json_inputs = self.model_def['Inputs'] | self.model_def['States']
        model_inputs = list(json_inputs.keys())
        ## Initialize the dictionary containing the data
        if name in list(self._data.keys()):
            log.warning(f'Dataset named {name} already loaded! overriding the existing one..')
        self._data[name] = {}

        num_of_samples = {}
        if type(source) is str:  ## we have a directory path containing the files
            ## collect column indexes
            format_idx = {}
            idx = 0
            for item in format:
                if isinstance(item, tuple):
                    for key in item:
                        if key not in model_inputs:
                            idx += 1
                            break
                        n_cols = json_inputs[key]['dim']
                        format_idx[key] = (idx, idx + n_cols)
                    idx += n_cols
                else:
                    if item not in model_inputs:
                        idx += 1
                        continue
                    n_cols = json_inputs[item]['dim']
                    format_idx[item] = (idx, idx + n_cols)
                    idx += n_cols

            ## Initialize each input key
            for key in format_idx.keys():
                self._data[name][key] = []

            ## obtain the file names
            try:
                _, _, files = next(os.walk(source))
                files.sort()
            except StopIteration as e:
                check(False, StopIteration, f'ERROR: The path "{source}" does not exist!')
                return
            self._file_count = len(files)
            if self._file_count > 1:  ## Multifile
                self._multifile[name] = []

            ## Cycle through all the files
            for file in files:
                try:
                    ## read the csv
                    df = pd.read_csv(os.path.join(source, file), skiprows=skiplines, delimiter=delimiter, header=header)
                except:
                    log.warning(f'Cannot read file {os.path.join(source, file)}')
                    continue
                if self._file_count > 1:
                    self._multifile[name].append(
                        (self._multifile[name][-1] + (len(df) - self._max_n_samples + 1)) if self._multifile[name] else len(
                            df) - self._max_n_samples + 1)
                ## Cycle through all the windows
                for key, idxs in format_idx.items():
                    back, forw = self._input_ns_backward[key], self._input_ns_forward[key]
                    ## Save as numpy array the data
                    data = df.iloc[:, idxs[0]:idxs[1]].to_numpy()
                    self._data[name][key] += [data[i - back:i + forw] for i in
                                              range(self._max_samples_backward, len(df) - self._max_samples_forward + 1)]

            ## Stack the files
            for key in format_idx.keys():
                self._data[name][key] = np.stack(self._data[name][key])
                num_of_samples[key] = self._data[name][key].shape[0]

        elif type(source) is dict:  ## we have a crafted dataset
            self._file_count = 1

            ## Check if the inputs are correct
            # assert set(model_inputs).issubset(source.keys()), f'The dataset is missing some inputs. Inputs needed for the model: {model_inputs}'

            # Merge a list of inputs into a single dictionary
            for key in model_inputs:
                if key not in source.keys():
                    continue

                self._data[name][key] = []  ## Initialize the dataset

                back, forw = self._input_ns_backward[key], self._input_ns_forward[key]
                for idx in range(len(source[key]) - self._max_n_samples + 1):
                    self._data[name][key].append(
                        source[key][idx + (self._max_samples_backward - back):idx + (self._max_samples_backward + forw)])

            ## Stack the files
            for key in model_inputs:
                if key not in source.keys():
                    continue
                self._data[name][key] = np.stack(self._data[name][key])
                if self._data[name][key].ndim == 2:  ## Add the sample dimension
                    self._data[name][key] = np.expand_dims(self._data[name][key], axis=-1)
                if self._data[name][key].ndim > 3:
                    self._data[name][key] = np.squeeze(self._data[name][key], axis=1)
                num_of_samples[key] = self._data[name][key].shape[0]

        elif isinstance(source, pd.DataFrame):  ## we have a crafted dataset
            self._file_count = 1

            ## Resampling if the time column is provided (must be a Datetime object)
            if resampling:
                if type(source.index) is pd.DatetimeIndex:
                    source = source.resample(f"{int(self.model_def.getSampleTime()  * 1e9)}ns").interpolate(method="linear")
                elif 'time' in source.columns:
                    if not ptypes.is_datetime64_any_dtype(source['time']):
                        source['time'] = pd.to_datetime(source['time'], unit='s')
                    source = source.set_index('time', drop=True)
                    source = source.resample(f"{int(self.model_def.getSampleTime() * 1e9)}ns").interpolate(method="linear")
                else:
                    raise TypeError(
                        "No time column found in the DataFrame. Please provide a time column for resampling.")

            processed_data = {}
            for key in model_inputs:
                if key not in source.columns:
                    continue

                processed_data[key] = []  ## Initialize the dataset
                back, forw = self._input_ns_backward[key], self._input_ns_forward[key]

                for idx in range(len(source) - self._max_n_samples + 1):
                    window = source[key].iloc[idx + (self._max_samples_backward - back):idx + (self._max_samples_backward + forw)]
                    processed_data[key].append(window.to_numpy())

            ## Convert lists to numpy arrays
            for key in processed_data:
                processed_data[key] = np.stack(processed_data[key])
                if json_inputs[key]['dim'] > 1:
                    processed_data[key] = np.array(processed_data[key].tolist(), dtype=np.float64)
                if processed_data[key].ndim == 2:  ## Add the sample dimension
                    processed_data[key] = np.expand_dims(processed_data[key], axis=-1)
                if processed_data[key].ndim > 3:
                    processed_data[key] = np.squeeze(processed_data[key], axis=1)
                num_of_samples[key] = processed_data[key].shape[0]

            self._data[name] = processed_data

        # Check dim of the samples
        check(len(set(num_of_samples.values())) == 1, ValueError,
              f"The number of the sample of the dataset {name} are not the same for all input in the dataset: {num_of_samples}")
        self._num_of_samples[name] = num_of_samples[list(num_of_samples.keys())[0]]

        ## Set the Loaded flag to True
        self._data_loaded = True
        ## Update the number of datasets loaded
        self.__n_datasets = len(self._data.keys())
        self.__datasets_loaded.add(name)
        ## Show the dataset
        self.visualizer.showDataset(name=name)
