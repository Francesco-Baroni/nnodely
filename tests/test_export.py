import sys, os, unittest, torch, shutil
import numpy as np

from nnodely import *
from nnodely import relation
relation.CHECK_NAMES = False

import torch.onnx
import onnx
import onnxruntime as ort
import importlib

from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

# 11 Tests
# Test of export and import the network to a file in different format

class ModelyExportTest(unittest.TestCase):

    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:
            self.assertEqual(len(data1),len(data2))
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def __init__(self, *args, **kwargs):
        super(ModelyExportTest, self).__init__(*args, **kwargs)

        self.result_path = './results'
        self.test = Modely(visualizer=None, seed=42, workspace=self.result_path)

        x = Input('x')
        y = Input('y')
        z = Input('z')

        ## create the relations
        def myFun(K1, p1, p2):
            return K1 * p1 * p2

        K_x = Parameter('k_x', dimensions=1, tw=1, init=init_constant, init_params={'value': 1})
        K_y = Parameter('k_y', dimensions=1, tw=1)
        w = Parameter('w', dimensions=1, tw=1, init=init_constant, init_params={'value': 1})
        t = Parameter('t', dimensions=1, tw=1)
        c_v = Constant('c_v', tw=1, values=[[1], [2]])
        c = 5
        w_5 = Parameter('w_5', dimensions=1, tw=5)
        t_5 = Parameter('t_5', dimensions=1, tw=5)
        c_5 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        parfun_x = ParamFun(myFun, parameters=[K_x], constants=[c_v])
        parfun_y = ParamFun(myFun, parameters=[K_y])
        parfun_z = ParamFun(myFun)
        fir_w = Fir(parameter=w_5)(x.tw(5))
        fir_t = Fir(parameter=t_5)(y.tw(5))
        time_part = TimePart(x.tw(5), i=1, j=3)
        sample_select = SampleSelect(x.sw(5), i=1)

        def fuzzyfun(x):
            return torch.tan(x)

        fuzzy = Fuzzify(output_dimension=4, range=[0, 4], functions=fuzzyfun)(x.tw(1))
        fuzzyTriang = Fuzzify(centers=[1, 2, 3, 7])(x.tw(1))

        out = Output('out', Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v)))
        # out = Output('out', Fir(parfun_x(x.tw(1))+parfun_y(y.tw(1),c_v)+parfun_z(x.tw(5),t_5,c_5)))
        out2 = Output('out2', Add(w, x.tw(1)) + Add(t, y.tw(1)) + Add(w, c))
        out3 = Output('out3', Add(fir_w, fir_t))
        out4 = Output('out4', Linear(output_dimension=1)(fuzzy+fuzzyTriang))
        out5 = Output('out5', Fir(time_part) + Fir(sample_select))
        out6 = Output('out6', LocalModel(output_function=Fir())(x.tw(1), fuzzy))

        self.test.addModel('modelA', out)
        self.test.addModel('modelB', [out2, out3, out4])
        self.test.addModel('modelC', [out4, out5, out6])
        self.test.addMinimize('error1', x.last(), out)
        self.test.addMinimize('error2', y.last(), out3, loss_function='rmse')
        self.test.addMinimize('error3', z.last(), out6, loss_function='rmse')

    def test_export_pt(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export torch file .pt
        # Save torch model and load it
        self.test.neuralizeModel(0.5)
        old_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.saveTorchModel()
        self.test.neuralizeModel(clear_model=True)
        # The new_out is different from the old_out because the model is cleared
        new_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        # The new_out_after_load is the same as the old_out because the model is loaded with the same parameters
        self.test.loadTorchModel()
        new_out_after_load = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})

        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out)
        self.assertEqual(old_out, new_out_after_load)

        with self.assertRaises(RuntimeError):
            test2 = Modely(visualizer=None, workspace = self.result_path)
            # You need not neuralized model to load a torch model
            test2.loadTorchModel()

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_json_not_neuralized(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export json of nnodely model before neuralize
        # Save a not neuralized nnodely json model and load it
        self.test.saveModel()  # Save a model without parameter values and samples values
        with self.assertRaises(RuntimeError):
            self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.loadModel()  # Load the nnodely model without parameter values
        with self.assertRaises(RuntimeError):
            self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        test2 = Modely(visualizer=None, workspace=self.test.getWorkspace())
        test2.loadModel()  # Load the nnodely model with parameter values
        self.assertEqual(test2.model_def.json, self.test.model_def.json)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_json_untrained(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export json of nnodely model
        # Save a untrained nnodely json model and load it
        # the new_out and new_out_after_load are different because the model saved model is not trained
        self.test.neuralizeModel(0.5)
        old_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.saveModel()  # Save a model without parameter values
        self.test.neuralizeModel(clear_model=True)  # Create a new torch model
        new_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.loadModel()  # Load the nnodely model without parameter values
        # Use the preloaded torch model for inference
        with self.assertRaises(RuntimeError):
            self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.neuralizeModel(0.5)
        new_out_after_load = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out)
        with self.assertRaises(AssertionError):
            self.assertEqual(new_out, new_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_json_trained(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export json of nnodely model with parameter valuess
        # The old_out is the same as the new_out_after_load because the model is loaded with the same parameters
        self.test.neuralizeModel(0.5)
        old_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.neuralizeModel()  # Load the parameter from torch model to nnodely model json
        self.test.saveModel()  # Save the model with and without parameter values
        self.test.neuralizeModel(clear_model=True)  # Create a new torch model
        new_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.loadModel()  # Load the nnodely model with parameter values
        with self.assertRaises(RuntimeError):
            self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.neuralizeModel()
        new_out_after_load = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out)
        with self.assertRaises(AssertionError):
            self.assertEqual(new_out, new_out_after_load)
        self.assertEqual(old_out, new_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_import_json_new_object(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Import nnodely json model in a new object
        self.test.neuralizeModel(0.5)
        old_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.neuralizeModel()
        self.test.saveModel()  # Save the model with and without parameter values
        test2 = Modely(visualizer=None, workspace=self.test.getWorkspace())
        test2.loadModel()  # Load the nnodely model with parameter values
        with self.assertRaises(RuntimeError):
            test2({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        test2.neuralizeModel()
        new_model_out_after_load = test2({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.assertEqual(old_out, new_model_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_torch_script(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export and import of a torch script .py
        # The old_out is the same as the new_out_after_load because the model is loaded with the same parameters
        with self.assertRaises(RuntimeError):
            self.test.exportPythonModel() # The model is not neuralized yet
        self.test.neuralizeModel(0.5)
        old_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.exportPythonModel()  # Export the trace model
        self.test.neuralizeModel(clear_model=True)  # Create a new torch model
        new_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.importPythonModel()  # Import the tracer model
        with self.assertRaises(RuntimeError):
            self.test.exportPythonModel() # The model is traced
        # Perform inference with the imported tracer model
        new_out_after_load = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        with self.assertRaises(AssertionError):
             self.assertEqual(old_out, new_out)
        self.assertEqual(old_out, new_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_torch_script_new_object(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Import of a torch script .py
        self.test.neuralizeModel(0.5,clear_model=True)
        old_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.exportPythonModel()  # Export the trace model
        self.test.neuralizeModel(clear_model=True)
        test2 = Modely(visualizer=None, workspace=self.test.getWorkspace())
        test2.importPythonModel()  # Load the nnodely model with parameter values
        with self.assertRaises(RuntimeError):
            test2.exportPythonModel() # The model is traced
        new_out_after_load = test2({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.assertEqual(old_out, new_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_trained_torch_script(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Perform training on an imported tracer model
        data_x = np.arange(0.0, 1, 0.1)
        data_y = np.arange(0.0, 1, 0.1)
        a, b = -1.0, 2.0
        dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
        params = {'num_of_epochs': 1, 'lr': 0.01}
        self.test.neuralizeModel(0.5,clear_model=True)
        old_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.exportPythonModel()  # Export the trace model
        self.test.loadData(name='dataset', source=dataset)  # Create the dataset
        self.test.trainModel(optimizer='SGD', training_params=params)  # Train the traced model
        new_out_after_train = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        with self.assertRaises(AssertionError):
             self.assertEqual(old_out, new_out_after_train)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_torch_script_new_object_train(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Perform training on an imported new tracer model
        self.test.neuralizeModel(0.5, clear_model=True)
        old_out = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        self.test.exportPythonModel()  # Export the trace model
        data_x = np.arange(0.0, 1, 0.1)
        data_y = np.arange(0.0, 1, 0.1)
        a, b = -1.0, 2.0
        dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
        params = {'num_of_epochs': 1, 'lr': 0.01}
        self.test.loadData(name='dataset', source=dataset)  # Create the dataset
        self.test.trainModel(optimizer='SGD', training_params=params)  # Train the traced model
        old_out_after_train = self.test({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        with self.assertRaises(AssertionError):
             self.assertEqual(old_out, old_out_after_train)
        test2 = Modely(visualizer=None, workspace=self.test.getWorkspace())
        test2.importPythonModel()  # Load the nnodely model with parameter values
        test2.loadData(name='dataset', source=dataset)  # Create the dataset
        test2.trainModel(optimizer='SGD', training_params=params)  # Train the traced model
        new_out_after_train = test2({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        with self.assertRaises(AssertionError):
             self.assertEqual(old_out, new_out_after_train)
        self.assertEqual(old_out_after_train, new_out_after_train)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_onnx(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)

        self.test.neuralizeModel(0.5, clear_model=True)
        # Export the all models in onnx format
        self.test.exportONNX(['x', 'y'], ['out', 'out2', 'out3', 'out4', 'out5', 'out6'])  # Export the onnx model
        # Export only the modelB in onnx format
        self.test.exportONNX(['x', 'y'], ['out3', 'out4', 'out2'], ['modelB'])  # Export the onnx model
        self.assertTrue(os.path.exists(os.path.join(self.test.getWorkspace(), 'onnx', 'net.onnx')))
        self.assertTrue(os.path.exists(os.path.join(self.test.getWorkspace(), 'onnx', 'net_modelB.onnx')))

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_report(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)

        self.test.resetSeed(42)
        self.test.neuralizeModel(0.5, clear_model=True)
        data_x = np.arange(0.0, 10, 0.1)
        data_y = np.arange(0.0, 10, 0.1)
        a, b = -1.0, 2.0
        dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
        params = {'num_of_epochs': 20, 'lr': 0.01}
        self.test.loadData(name='dataset', source=dataset)  # Create the dataset
        self.test.trainModel(optimizer='SGD', training_params=params)  # Train the traced model
        self.test.exportReport()

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_and_import_train_python_module(self):
        result_path = './results'
        test = Modely(visualizer=None, seed=42, workspace=result_path)
        x = Input('x')
        y = State('y')
        z = State('z')
        target = Input('target')
        a = Parameter('a', dimensions=1, sw=1, values=[[1]])
        b = Parameter('b', dimensions=1, sw=1, values=[[1]])
        c = Parameter('c', dimensions=1, sw=1, values=[[1]])
        fir_x = Fir(parameter=a)(x.last())
        fir_y = Fir(parameter=b)(y.last())
        fir_z = Fir(parameter=c)(z.last())
        data_x, data_y, data_z = np.random.rand(20), np.random.rand(20), np.random.rand(20)
        dataset = {'x':data_x, 'y':data_y, 'z':data_z, 'target':3*data_x + 3*data_y + 3*data_z}
        fir_x.connect(y)
        sum_rel = fir_x + fir_y + fir_z
        sum_rel.closedLoop(z)
        out = Output('out', sum_rel)
        test.addModel('model', out)
        test.addMinimize('error', target.last(), out)
        test.neuralizeModel(0.5)
        test.loadData(name='test_dataset', source=dataset)
        ## Train
        test.trainModel(optimizer='SGD', training_params={'num_of_epochs': 1, 'lr': 0.0001, 'train_batch_size': 1}, splits=[100,0,0], prediction_samples=10)
        ## Inference
        sample = {'x':[1], 'y':[2], 'z':[3], 'target':[18]}
        train_result = test(sample)
        train_parameters = test.model.all_parameters
        # Export the model
        test.exportPythonModel()
        # Import the model
        test.importPythonModel(name='net')
        # Inference with imported model
        self.assertEqual(train_result, test(sample))
        self.assertEqual(train_parameters['a'], test.model.all_parameters['a'])
        self.assertEqual(train_parameters['b'], test.model.all_parameters['b'])
        self.assertEqual(train_parameters['c'], test.model.all_parameters['c'])

    def test_export_and_import_python_module(self):
        result_path = './results'
        test = Modely(visualizer=None, seed=42, workspace=result_path)
        x = Input('x')
        y = State('y')
        z = State('z')
        a = Parameter('a', dimensions=1, sw=1, values=[[1]])
        b = Parameter('b', dimensions=1, sw=1, values=[[1]])
        c = Parameter('c', dimensions=1, sw=1, values=[[1]])
        fir_x = Fir(parameter=a)(x.last())
        fir_y = Fir(parameter=b)(y.last())
        fir_z = Fir(parameter=c)(z.last())
        fir_x.connect(y)
        sum_rel = fir_x + fir_y + fir_z
        sum_rel.closedLoop(z)
        out = Output('out', sum_rel)
        test.addModel('model', out)
        test.neuralizeModel(0.5)
        ## Inference
        sample = {'x':[1], 'y':[2], 'z':[3]}
        inference_result = test(sample)
        self.assertEqual(inference_result['out'], [5.0])
        # Export the model
        test.exportPythonModel(name='exported_model')
        ## Load the exported model.py
        model_folder = 'results'
        model_filename = 'exported_model.py'
        model_path = os.path.join(model_folder, model_filename)
        sys.path.insert(0, model_folder)
        ## Import the python exported module
        module_name = os.path.splitext(model_filename)[0]
        module = importlib.import_module(module_name)
        RecurrentModel = getattr(module, 'RecurrentModel')
        model = RecurrentModel()
        model.eval()
        # Create dummy input data
        dummy_input = {'x': torch.ones(5, 1, 1, 1), 'target': torch.ones(10, 1, 1, 1), 'y': torch.zeros(1,1,1), 'z':torch.zeros(1,1,1)}  # Adjust the shape as needed
        # Inference with imported model
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output['out'], [torch.tensor([[[2.]]]), torch.tensor([[[4.]]]), torch.tensor([[[6.]]]), torch.tensor([[[8.]]]), torch.tensor([[[10.]]])])

    def test_export_and_import_onnx_module(self):
        result_path = './results'
        test = Modely(visualizer=None, seed=42, workspace=result_path)
        x = Input('x')
        y = State('y')
        z = State('z')
        a = Parameter('a', dimensions=1, sw=1, values=[[1]])
        b = Parameter('b', dimensions=1, sw=1, values=[[1]])
        c = Parameter('c', dimensions=1, sw=1, values=[[1]])
        fir_x = Fir(parameter=a)(x.last())
        fir_y = Fir(parameter=b)(y.last())
        fir_z = Fir(parameter=c)(z.last())
        fir_x.connect(y)
        sum_rel = fir_x + fir_y + fir_z
        sum_rel.closedLoop(z)
        out = Output('out', sum_rel)
        test.addModel('model', out)
        test.neuralizeModel(0.5)
        ## Inference
        sample = {'x':[1], 'y':[2], 'z':[3]}
        inference_result = test(sample)
        self.assertEqual(inference_result['out'], [5.0])
        ## Export in ONNX format
        test.exportONNX(['x','y','z'],['out']) # Export the onnx model

        ## ONNX IMPORT
        onnx_model_path = os.path.join('results', 'onnx', 'net.onnx')
        dummy_input = {'x':np.ones(shape=(3, 1, 1, 1)).astype(np.float32),
                       'y':np.ones(shape=(1, 1, 1)).astype(np.float32),
                       'z':np.ones(shape=(1, 1, 1)).astype(np.float32)}
        outputs = Modely().onnxInference(dummy_input,onnx_model_path)
        # Get the output
        expected_output = np.array([[[[3.]]], [[[5.]]], [[[7.]]]], dtype=np.float32)
        self.assertEqual(outputs[0].tolist(), expected_output.tolist())

    def test_export_and_import_onnx_module_easy(self):
        result_path = './results'
        test = Modely(visualizer=None, seed=42, workspace=result_path)
        num_cycle = Input('num_cycle')
        x = State('x')
        fir_x = Fir()(x.last()+1.0)
        fir_x.closedLoop(x)
        out1 = Output('out1', fir_x)
        out2 = Output('out2', num_cycle.last()+1.0)
        test.addModel('model', [out1,out2])
        test.neuralizeModel(0.5)

        ## Export in ONNX format
        test.exportONNX(['x','num_cycle'],['out1','out2']) # Export the onnx model

        ## ONNX IMPORT
        onnx_model_path = os.path.join('results', 'onnx', 'net.onnx')
        outputs = Modely().onnxInference(inputs={'num_cycle':np.ones(shape=(10, 1, 1, 1)).astype(np.float32), 'x':np.ones(shape=(1, 1, 1)).astype(np.float32)}, path=onnx_model_path)

    def test_export_and_import_onnx_module_complex(self):
        # Create nnodely structure
        vehicle = Modely(visualizer=None, seed=2, workspace=os.path.join(os.getcwd(), 'results'))

        # Dimensions of the layers
        n  = 25
        na = 21

        #Create neural model inputs
        velocity = Input('vel')
        brake = Input('brk')
        gear = Input('gear')
        torque = Input('trq')
        altitude = Input('alt',dimensions=na)
        acc = Input('acc')

        # Create neural network relations
        air_drag_force = Linear(b=True)(velocity.last()**2)
        breaking_force = -Relu(Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.002, 'lambda':3})(brake.sw(n)))
        gravity_force = Linear(W_init=init_constant, W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.last())
        fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
        local_model = LocalModel(input_function=lambda: Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.002, 'lambda':3}))
        engine_force = local_model(torque.sw(n), fuzzi_gear)

        # Create neural network output
        out = Output('accelleration', air_drag_force+breaking_force+gravity_force+engine_force)

        # Add the neural model to the nnodely structure and neuralization of the model
        vehicle.addModel('acc',[out])
        vehicle.addMinimize('acc_error', acc.last(), out, loss_function='rmse')
        vehicle.neuralizeModel(0.05)

        # Load the training and the validation dataset
        data_struct = ['vel','trq','brk','gear','alt','acc']
        data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'vehicle_data')
        vehicle.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1)

        # Inference
        sample = vehicle.getSamples('dataset', window=1)
        model_inference = vehicle(sample, sampled=True)

        ## Export the Onnx Model
        vehicle.exportONNX(['vel','brk','gear','trq','alt'],['accelleration'])

        ## Onnx Import
        onnx_model_path = os.path.join('results', 'onnx', 'net.onnx')
        outputs = Modely().onnxInference(sample, onnx_model_path)
        self.assertEqual(outputs[0][0][0].tolist(), model_inference['accelleration'])

    def test_export_python_module_recurrent(self):
        test = Modely(visualizer=None, seed=42, workspace=os.path.join(os.getcwd(), 'results'))
        input1 = Input('input1')
        input2 = Input('input2', dimensions=3)
        input3 = Input('input3')
        input4 = Input('input4', dimensions=3)
        state1 = State('state1')
        state2 = State('state2', dimensions=3)

        rel_1 = Linear(b=True)(input1.last()) + Linear(b=True)(input3.last())
        rel_1.closedLoop(state1)

        rel_2 = Linear(output_dimension=3, b=True)(input2.last()) + Linear(output_dimension=3, b=True)(input4.last())
        rel_2.closedLoop(state2)

        out1 = Output('out1', rel_1)
        out2 = Output('out2', rel_2)
        out3 = Output('input1', input1.last())
        out4 = Output('input2', input2.last())
        out5 = Output('input3', input3.sw(4))
        out6 = Output('input4', input4.sw(4))
        out7 = Output('state1', state1.last())
        out8 = Output('state2', state2.last())

        test.addModel('model', [out1, out2, out3, out4, out5, out6, out7, out8])
        test.neuralizeModel()

        test.exportPythonModel(name='net')

        ## Load the exported model.py
        model_folder = 'results'
        model_filename = 'net.py'
        model_path = os.path.join(model_folder, model_filename)
        sys.path.insert(0, model_folder)
        ## Import the python exported module
        module_name = os.path.splitext(model_filename)[0]
        module = importlib.import_module(module_name)
        RecurrentModel = getattr(module, 'RecurrentModel')
        recurrent_model = RecurrentModel()
        recurrent_model.eval()

        ## Without Horizon and without batch
        recurrent_sample = {'input1': torch.rand(size=(1,1,1,1), dtype=torch.float32),
                            'input2': torch.rand(size=(1,1,1,3), dtype=torch.float32),
                            'input3': torch.rand(size=(1,1,4,1), dtype=torch.float32),
                            'input4': torch.rand(size=(1,1,4,3), dtype=torch.float32)}
        recurrent_sample['state1'] = torch.rand(size=(1,1,1), dtype=torch.float32)
        recurrent_sample['state2'] = torch.rand(size=(1,1,3), dtype=torch.float32)
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out1']).shape), [1,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out2']).shape), [1,1,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input1']).shape), [1,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input2']).shape), [1,1,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input3']).shape), [1,1,4,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input4']).shape), [1,1,4,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state1']).shape), [1,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state2']).shape), [1,1,1,3])

        ## With Horizon and without batch
        recurrent_sample = {'input1': torch.rand(size=(5,1,1,1), dtype=torch.float32),
                            'input2': torch.rand(size=(5,1,1,3), dtype=torch.float32),
                            'input3': torch.rand(size=(5,1,4,1), dtype=torch.float32),
                            'input4': torch.rand(size=(5,1,4,3), dtype=torch.float32)}
        recurrent_sample['state1'] = torch.rand(size=(1,1,1), dtype=torch.float32)
        recurrent_sample['state2'] = torch.rand(size=(1,1,3), dtype=torch.float32)
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out1']).shape), [5,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out2']).shape), [5,1,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input1']).shape), [5,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input2']).shape), [5,1,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input3']).shape), [5,1,4,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input4']).shape), [5,1,4,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state1']).shape), [5,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state2']).shape), [5,1,1,3])

        ## With Horizon and with batch
        recurrent_sample = {'input1': torch.rand(size=(5,2,1,1), dtype=torch.float32),
                            'input2': torch.rand(size=(5,2,1,3), dtype=torch.float32),
                            'input3': torch.rand(size=(5,2,4,1), dtype=torch.float32),
                            'input4': torch.rand(size=(5,2,4,3), dtype=torch.float32)}
        recurrent_sample['state1'] = torch.rand(size=(2,1,1), dtype=torch.float32)
        recurrent_sample['state2'] = torch.rand(size=(2,1,3), dtype=torch.float32)
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out1']).shape), [5,2,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out2']).shape), [5,2,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input1']).shape), [5,2,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input2']).shape), [5,2,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input3']).shape), [5,2,4,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input4']).shape), [5,2,4,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state1']).shape), [5,2,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state2']).shape), [5,2,1,3])

    def test_export_onnx_module_recurrent(self):
        test = Modely(visualizer=None, seed=42, workspace=os.path.join(os.getcwd(), 'results'))
        onnx_model_path = os.path.join('results', 'onnx', 'net.onnx')
        input1 = Input('input1')
        input2 = Input('input2', dimensions=3)
        input3 = Input('input3')
        input4 = Input('input4', dimensions=3)
        state1 = State('state1')
        state2 = State('state2', dimensions=3)

        rel_1 = Linear(b=True)(input1.last()) + Linear(b=True)(input3.last())
        rel_1.closedLoop(state1)

        rel_2 = Linear(output_dimension=3, b=True)(input2.last()) + Linear(output_dimension=3, b=True)(input4.last())
        rel_2.closedLoop(state2)

        out1 = Output('out1', rel_1)
        out2 = Output('out2', rel_2)
        out3 = Output('out_input1', input1.last())
        out4 = Output('out_input2', input2.last())
        out5 = Output('out_input3', input3.sw(4))
        out6 = Output('out_input4', input4.sw(4))
        out7 = Output('out_state1', state1.last())
        out8 = Output('out_state2', state2.last())

        test.addModel('model', [out1, out2, out3, out4, out5, out6, out7, out8])
        test.neuralizeModel()

        test.exportONNX(inputs_order=['input1','input2','input3','input4','state1','state2'],outputs_order=['out1', 'out2', 'out_input1', 'out_input2', 'out_input3', 'out_input4', 'out_state1', 'out_state2'])

        ## Without Horizon and without batch
        recurrent_sample = {'input1': np.random.rand(1,1,1,1).astype(np.float32),
                            'input2': np.random.rand(1,1,1,3).astype(np.float32),
                            'input3': np.random.rand(1,1,4,1).astype(np.float32),
                            'input4': np.random.rand(1,1,4,3).astype(np.float32)}
        recurrent_sample['state1'] = np.random.rand(1,1,1).astype(np.float32)
        recurrent_sample['state2'] = np.random.rand(1,1,3).astype(np.float32)
        inference = Modely().onnxInference(recurrent_sample, onnx_model_path)
        self.assertListEqual(list(inference[0].shape), [1,1,1,1])
        self.assertListEqual(list(inference[1].shape), [1,1,1,3])
        self.assertListEqual(list(inference[2].shape), [1,1,1,1])
        self.assertListEqual(list(inference[3].shape), [1,1,1,3])
        self.assertListEqual(list(inference[4].shape), [1,1,4,1])
        self.assertListEqual(list(inference[5].shape), [1,1,4,3])
        self.assertListEqual(list(inference[6].shape), [1,1,1,1])
        self.assertListEqual(list(inference[7].shape), [1,1,1,3])

        ## With Horizon and without batch
        recurrent_sample = {'input1': np.random.rand(5,1,1,1).astype(np.float32),
                            'input2': np.random.rand(5,1,1,3).astype(np.float32),
                            'input3': np.random.rand(5,1,4,1).astype(np.float32),
                            'input4': np.random.rand(5,1,4,3).astype(np.float32)}
        recurrent_sample['state1'] = np.random.rand(1,1,1).astype(np.float32)
        recurrent_sample['state2'] = np.random.rand(1,1,3).astype(np.float32)
        inference = Modely().onnxInference(recurrent_sample, onnx_model_path)
        self.assertListEqual(list(inference[0].shape), [5,1,1,1])
        self.assertListEqual(list(inference[1].shape), [5,1,1,3])
        self.assertListEqual(list(inference[2].shape), [5,1,1,1])
        self.assertListEqual(list(inference[3].shape), [5,1,1,3])
        self.assertListEqual(list(inference[4].shape), [5,1,4,1])
        self.assertListEqual(list(inference[5].shape), [5,1,4,3])
        self.assertListEqual(list(inference[6].shape), [5,1,1,1])
        self.assertListEqual(list(inference[7].shape), [5,1,1,3])

        # ## With Horizon and with batch
        recurrent_sample = {'input1': np.random.rand(5,2,1,1).astype(np.float32),
                            'input2': np.random.rand(5,2,1,3).astype(np.float32),
                            'input3': np.random.rand(5,2,4,1).astype(np.float32),
                            'input4': np.random.rand(5,2,4,3).astype(np.float32)}
        recurrent_sample['state1'] = np.random.rand(2,1,1).astype(np.float32)
        recurrent_sample['state2'] = np.random.rand(2,1,3).astype(np.float32)
        inference = Modely().onnxInference(recurrent_sample, onnx_model_path)
        self.assertListEqual(list(inference[0].shape), [5,2,1,1])
        self.assertListEqual(list(inference[1].shape), [5,2,1,3])
        self.assertListEqual(list(inference[2].shape), [5,2,1,1])
        self.assertListEqual(list(inference[3].shape), [5,2,1,3])
        self.assertListEqual(list(inference[4].shape), [5,2,4,1])
        self.assertListEqual(list(inference[5].shape), [5,2,4,3])
        self.assertListEqual(list(inference[6].shape), [5,2,1,1])
        self.assertListEqual(list(inference[7].shape), [5,2,1,3])
        

    def test_export_and_import_python_module_complex_recurrent(self):
        # Create nnodely structure
        vehicle = Modely(visualizer=None, seed=2, workspace=os.path.join(os.getcwd(), 'results'))

        # Dimensions of the layers
        n  = 25
        na = 21

        #Create neural model inputs
        velocity = State('vel')
        brake = Input('brk')
        gear = Input('gear')
        torque = Input('trq')
        altitude = Input('alt',dimensions=na)
        acc = Input('acc')

        # Create neural network relations
        air_drag_force = Linear(b=True)(velocity.last()**2)
        breaking_force = -Relu(Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.002, 'lambda':3})(brake.sw(n)))
        gravity_force = Linear(W_init=init_constant, W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.last())
        fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
        local_model = LocalModel(input_function=lambda: Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.002, 'lambda':3}))
        engine_force = local_model(torque.sw(n), fuzzi_gear)

        sum_rel = air_drag_force+breaking_force+gravity_force+engine_force
        sum_rel.closedLoop(velocity)

        # Create neural network output
        out = Output('accelleration', sum_rel)

        # Add the neural model to the nnodely structure and neuralization of the model
        vehicle.addModel('acc',[out])
        vehicle.addMinimize('acc_error', acc.last(), out, loss_function='rmse')
        vehicle.neuralizeModel(0.05)

        # Load the training and the validation dataset
        data_struct = ['vel','trq','brk','gear','alt','acc']
        data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'vehicle_data')
        vehicle.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1)

        # Inference
        sample = vehicle.getSamples('dataset', window=3)
        model_inference = vehicle(sample, sampled=True, prediction_samples=3)

        vehicle.exportPythonModel(name='net')

        vehicle.importPythonModel(name='net')
        model_import_inference = vehicle(sample, sampled=True, prediction_samples=3)
        self.assertEqual(model_inference['accelleration'], model_import_inference['accelleration'])

        ## Load the exported model.py
        model_folder = 'results'
        model_filename = 'net.py'
        model_path = os.path.join(model_folder, model_filename)
        sys.path.insert(0, model_folder)
        ## Import the python exported module
        module_name = os.path.splitext(model_filename)[0]
        module = importlib.import_module(module_name)
        RecurrentModel = getattr(module, 'RecurrentModel')
        recurrent_model = RecurrentModel()
        recurrent_model.eval()

        sample = vehicle.getSamples('dataset', window=3)
        recurrent_sample = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(1) for key, value in sample.items()}
        recurrent_sample['vel'] = torch.zeros(1,1,1)
        model_sample = {key: value for key, value in sample.items() if key != 'vel'}
        self.TestAlmostEqual([item.detach().item() for item in recurrent_model(recurrent_sample)['accelleration']], vehicle(model_sample, sampled=True, prediction_samples=3)['accelleration'])

    def test_export_and_import_onnx_module_complex_recurrent(self):
        # Create nnodely structure
        vehicle = Modely(visualizer=None, seed=42, workspace=os.path.join(os.getcwd(), 'results'))

        # Dimensions of the layers
        n  = 25
        na = 21

        #Create neural model inputs
        velocity = State('vel')
        brake = Input('brk')
        gear = Input('gear')
        torque = Input('trq')
        altitude = Input('alt',dimensions=na)
        acc = Input('acc')

        # Create neural network relations
        air_drag_force = Linear(b=True)(velocity.last()**2)
        breaking_force = -Relu(Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.002, 'lambda':3})(brake.sw(n)))
        gravity_force = Linear(W_init=init_constant, W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.last())
        fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
        local_model = LocalModel(input_function=lambda: Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.002, 'lambda':3}))
        engine_force = local_model(torque.sw(n), fuzzi_gear)

        sum_rel = air_drag_force+breaking_force+gravity_force+engine_force
        sum_rel.closedLoop(velocity)

        # Create neural network output
        out = Output('accelleration', sum_rel)

        # Add the neural model to the nnodely structure and neuralization of the model
        vehicle.addModel('acc',[out])
        vehicle.addMinimize('acc_error', acc.last(), out, loss_function='rmse')
        vehicle.neuralizeModel(0.05)

        # Load the training and the validation dataset
        data_struct = ['vel','trq','brk','gear','alt','acc']
        data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'vehicle_data')
        vehicle.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1)

        ## Export the Onnx Model
        vehicle.exportONNX(inputs_order=['gear','trq','alt','brk','vel'],outputs_order=['accelleration'])

        model_sample = vehicle.getSamples('dataset', window=1)
        model_inference = vehicle(model_sample, sampled=True, prediction_samples=1)

        ## ONNX IMPORT
        onnx_model_path = os.path.join('results', 'onnx', 'net.onnx')
        onnx_sample = {key: (np.expand_dims(value, axis=1).astype(np.float32) if key != 'vel' else value)  for key, value in model_sample.items()}
        outputs = Modely().onnxInference(onnx_sample, onnx_model_path)
        self.assertEqual(outputs[0][0], model_inference['accelleration'])

        model_sample = vehicle.getSamples('dataset', window=3)
        onnx_sample = {key: (np.expand_dims(value, axis=1).astype(np.float32) if key != 'vel' else np.expand_dims(np.array(value[0], dtype=np.float32), axis=0))  for key, value in model_sample.items()}
        model_inference = vehicle(model_sample, sampled=True, prediction_samples=3)
        outputs = Modely().onnxInference(onnx_sample, onnx_model_path)
        self.assertEqual(outputs[0].squeeze().tolist(), model_inference['accelleration'])


if __name__ == '__main__':
    unittest.main()


