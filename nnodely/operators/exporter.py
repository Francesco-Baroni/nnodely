from nnodely.exporter.standardexporter import StandardExporter
from nnodely.exporter.emptyexporter import EmptyExporter
from nnodely.model import Model
from nnodely.modeldef import ModelDef
from nnodely.utils import check

class Exporter:
    def __init__(self, exporter, workspace, save_history):
        check(type(self) is not Exporter, TypeError, "Exporter class cannot be instantiated directly")

        # Exporter
        if exporter == 'Standard':
            self.exporter = StandardExporter(workspace, self.visualizer, save_history)
        elif exporter != None:
            self.exporter = exporter
        else:
            self.exporter = EmptyExporter()

    def getWorkspace(self):
        return self.exporter.getWorkspace()

    def saveTorchModel(self, name='net', model_folder=None, models=None):
        """
        Saves the neural network model in PyTorch format.

        Parameters
        ----------
        name : str, optional
            The name of the saved model file. Default is 'net'.
        model_folder : str or None, optional
            The folder to save the model file in. Default is None.
        models : list or None, optional
            A list of model names to save. If None, the entire model is saved. Default is None.

        Raises
        ------
        RuntimeError
            If the model is not neuralized.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.neuralizeModel()
            >>> model.saveTorchModel(name='example_model', model_folder='path/to/save')
        """
        check(self.neuralized == True, RuntimeError, 'The model is not neuralized yet!')
        if models is not None:
            if name == 'net':
                name += '_' + '_'.join(models)
            model_def = ModelDef()
            model_def.update(model_dict={key: self.model_dict[key] for key in models if key in self.model_dict})
            model_def.setBuildWindow(self.model_def['Info']['SampleTime'])
            model_def.updateParameters(self.model)
            model = Model(model_def.json)
        else:
            model = self.model
        self.exporter.saveTorchModel(model, name, model_folder)

    def loadTorchModel(self, name='net', model_folder=None):
        """
        Loads a neural network model from a PyTorch format file.

        Parameters
        ----------
        name : str, optional
            The name of the model file to load. Default is 'net'.
        model_folder : str or None, optional
            The folder to load the model file from. Default is None.

        Raises
        ------
        RuntimeError
            If the model is not neuralized.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.neuralizeModel()
            >>> model.loadTorchModel(name='example_model', model_folder='path/to/load')
        """
        check(self.neuralized == True, RuntimeError, 'The model is not neuralized yet.')
        self.exporter.loadTorchModel(self.model, name, model_folder)

    def saveModel(self, name='net', model_path=None, models=None):
        ## TODO: Add tests passing the attribute 'models'
        """
        Saves the neural network model definition in a json file.

        Parameters
        ----------
        name : str, optional
            The name of the saved model file. Default is 'net'.
        model_path : str or None, optional
            The path to save the model file. Default is None.
        models : list or None, optional
            A list of model names to save. If None, the entire model is saved. Default is None.

        Raises
        ------
        RuntimeError
            If the network has not been defined.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.neuralizeModel()
            >>> model.saveModel(name='example_model', model_path='path/to/save')
        """
        if models is not None:
            if name == 'net':
                name += '_' + '_'.join(models)
            model_def = ModelDef()
            model_def.update(
                model_dict={key: self.model_def.model_dict[key] for key in models if key in self.model_def.model_dict})
            model_def.setBuildWindow(self.model_def['Info']['SampleTime'])
            model_def.updateParameters(self.model)
        else:
            model_def = self.model_def
        check(model_def.isDefined(), RuntimeError, "The network has not been defined.")
        self.exporter.saveModel(model_def.json, name, model_path)

    def loadModel(self, name=None, model_folder=None):
        """
        Loads a neural network model from a json file containing the model definition.

        Parameters
        ----------
        name : str or None, optional
            The name of the model file to load. Default is 'net'.
        model_folder : str or None, optional
            The folder to load the model file from. Default is None.

        Raises
        ------
        RuntimeError
            If there is an error loading the network.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.loadModel(name='example_model', model_folder='path/to/load')
        """
        if name is None:
            name = 'net'
        model_def = self.exporter.loadModel(name, model_folder)
        check(model_def, RuntimeError, "Error to load the network.")
        self.model_def = ModelDef(model_def)
        self.model = None
        self.neuralized = False
        self.traced = False

    def exportPythonModel(self, name='net', model_path=None, models=None):
        """
        Exports the neural network model as a standalone PyTorch Module class.

        Parameters
        ----------
        name : str, optional
            The name of the exported model file. Default is 'net'.
        model_path : str or None, optional
            The path to save the exported model file. Default is None.
        models : list or None, optional
            A list of model names to export. If None, the entire model is exported. Default is None.

        Raises
        ------
        RuntimeError
            If the network has not been defined.
            If the model is traced and cannot be exported to Python.
            If the model is not neuralized.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely(name='example_model')
            >>> model.neuralizeModel()
            >>> model.exportPythonModel(name='example_model', model_path='path/to/export')
        """
        if models is not None:
            if name == 'net':
                name += '_' + '_'.join(models)
            model_def = ModelDef()
            model_def.update(
                model_dict={key: self.model_def.model_dict[key] for key in models if key in self.model_def.model_dict})
            model_def.setBuildWindow(self.model_def['Info']['SampleTime'])
            model_def.updateParameters(self.model)
            model = Model(model_def.json)
        else:
            model_def = self.model_def
            model = self.model
        # check(model_def['States'] == {}, TypeError, "The network has state variables. The export to python is not possible.")
        check(model_def.isDefined(), RuntimeError, "The network has not been defined.")
        check(self.traced == False, RuntimeError,
              'The model is traced and cannot be exported to Python.\n Run neuralizeModel() to recreate a standard model.')
        check(self.neuralized == True, RuntimeError, 'The model is not neuralized yet.')
        self.exporter.saveModel(model_def.json, name, model_path)
        self.exporter.exportPythonModel(model_def, model, name, model_path)

    def importPythonModel(self, name=None, model_folder=None):
        """
        Imports a neural network model from a standalone PyTorch Module class.

        Parameters
        ----------
        name : str or None, optional
            The name of the model file to import. Default is 'net'.
        model_folder : str or None, optional
            The folder to import the model file from. Default is None.

        Raises
        ------
        RuntimeError
            If there is an error loading the network.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.importPythonModel(name='example_model', model_folder='path/to/import')
        """
        if name is None:
            name = 'net'
        model_def = self.exporter.loadModel(name, model_folder)
        check(model_def is not None, RuntimeError, "Error to load the network.")
        self.neuralizeModel(model_def=model_def)
        self.model = self.exporter.importPythonModel(name, model_folder)
        self.traced = True
        self.model_def.updateParameters(self.model)

    def exportONNX(self, inputs_order=None, outputs_order=None, models=None, name='net', model_folder=None):
        """
        Exports the neural network model to an ONNX file.

        .. note::
            The input_order may contain all the inputs and states of the model in the order that you want to export them.

        Parameters
        ----------
        inputs_order : list
            The order of the input and state variables.
        outputs_order : list
            The order of the output variables.
        models : list or None, optional
            A list of model names to export. If None, the entire model is exported. Default is None.
        name : str, optional
            The name of the exported ONNX file. Default is 'net'.
        model_folder : str or None, optional
            The folder to save the exported ONNX file. Default is None.

        Raises
        ------
        RuntimeError
            If the network has not been defined.
            If the model is traced and cannot be exported to ONNX.
            If the model is not neuralized.
            If the model is loaded and not created.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example usage:
            >>> input1 = Input('input1').last()
            >>> input2 = Input('input2').last()
            >>> out = Output('output1', input1+input2)

            >>> model = Modely()
            >>> model.neuralizeModel()
            >>> model.exportONNX(inputs_order=['input1', 'input2'], outputs_order=['output1'], name='example_model', model_folder='path/to/export')
        """
        check(self.model_def.isDefined(), RuntimeError, "The network has not been defined.")
        check(self.traced == False, RuntimeError,
              'The model is traced and cannot be exported to ONNX.\n Run neuralizeModel() to recreate a standard model.')
        check(self.neuralized == True, RuntimeError, 'The model is not neuralized yet.')
        check(self.model_def.model_dict != {}, RuntimeError, 'The model is loaded and not created.')
        model_def = ModelDef()
        if models is not None:
            if name == 'net':
                name += '_' + '_'.join(models)
            model_def.update(
                model_dict={key: self.model_def.model_dict[key] for key in models if key in self.model_def.model_dict})
        else:
            model_def.update(model_dict=self.model_def.model_dict)
        model_def.setBuildWindow(self.model_def['Info']['SampleTime'])
        model_def.updateParameters(self.model)
        model = Model(model_def.json)
        model.update()
        self.exporter.exportONNX(model_def, model, inputs_order, outputs_order, name, model_folder)

    def onnxInference(self, inputs: dict, path: str):
        """
        Run an inference session using an onnx model previously exported using the nnodely framework.

        .. note:: Feed-Forward ONNX model
            For feed-forward models, the onnx model expect all the inputs and states to have 3 dimensions. The first dimension is the batch size, the second is the time window and the third is the feature dimension.
        .. note:: Recurrent ONNX model
            For recurrent models, the onnx model expect all the inputs to have 4 dimensions. The first dimension is the prediction horizon, the second is the batch size, the third is the time window and the fourth is the feature dimension.
            For recurrent models, the onnx model expect all the States to have 3 dimensions. The first dimension is the batch size, the second is the time window, the third is the feature dimension

        Parameters
        ----------
        inputs : dict
            A dictionary containing the input and state variables to be used to make the inference.
            State variables are mandatory and are used to initialize the states of the model.
        path : str
            The path to the ONNX file to use.

        Raises
        ------
        RuntimeError
            If the shape of the inputs are not equals to the ones defined in the onnx model.
            If the batch size is not equal for all the inputs and states.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example - Feed-Forward:
            >>> x = Input('x')

            >>> onnx_model_path = path/to/net.onnx
            >>> dummy_input = {'x':np.ones(shape=(3, 1, 1)).astype(np.float32)}
            >>> predictions = Modely().onnxInference(dummy_input, onnx_model_path)
        Example - Recurrent:
            >>> x = Input('x')
            >>> y = State('y')

            >>> onnx_model_path = path/to/net.onnx
            >>> dummy_input = {'x':np.ones(shape=(3, 1, 1, 1)).astype(np.float32)
                                'y':np.ones(shape=(1, 1, 1)).astype(np.float32)}
            >>> predictions = Modely().onnxInference(dummy_input, onnx_model_path)
        """
        return self.exporter.onnxInference(inputs, path)

    def exportReport(self, name='net', model_folder=None):
        """
        Generates a PDF report with plots containing the results of the training and validation of the neural network.

        Parameters
        ----------
        name : str, optional
            The name of the exported report file. Default is 'net'.
        model_folder : str or None, optional
            The folder to save the exported report file. Default is None.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/export.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.neuralizeModel()
            >>> model.trainModel(train_dataset='train_dataset', validation_dataset='val_dataset', num_of_epochs=10)
            >>> model.exportReport(name='example_model', model_folder='path/to/export')
        """
        self.exporter.exportReport(self, name, model_folder)
