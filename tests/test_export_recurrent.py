import sys, os, unittest, torch, shutil, torch.onnx, importlib
import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj, Stream
from nnodely.support.logger import logging, nnLogger

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


    def test_export_and_import_train_python_module(self):
        NeuObj.clearNames()
        result_path = 'results'
        network_name = 'net'
        test = Modely(visualizer=None, seed=42, workspace=result_path)
        x = Input('x')
        y = State('y')
        z = State('z')
        target = Input('target')
        a = Parameter('a', dimensions=1, sw=1, values=[[1]])
        b = Parameter('b', dimensions=1, sw=1, values=[[1]])
        c = Parameter('c', dimensions=1, sw=1, values=[[1]])
        fir_x = Fir(W=a)(x.last())
        fir_y = Fir(W=b)(y.last())
        fir_z = Fir(W=c)(z.last())
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
        test.importPythonModel(name=network_name)
        # Inference with imported model
        self.assertEqual(train_result, test(sample))
        self.assertEqual(train_parameters['a'], test.model.all_parameters['a'])
        self.assertEqual(train_parameters['b'], test.model.all_parameters['b'])
        self.assertEqual(train_parameters['c'], test.model.all_parameters['c'])

        if os.path.exists(test.getWorkspace()):
            shutil.rmtree(test.getWorkspace())

    def test_export_and_import_python_module(self):
        NeuObj.clearNames()
        result_path = 'results'
        network_name = 'exported_model'
        test = Modely(visualizer=None, seed=42, workspace=result_path)
        x = Input('x')
        y = State('y')
        z = State('z')
        a = Parameter('a', dimensions=1, sw=1, values=[[1]])
        b = Parameter('b', dimensions=1, sw=1, values=[[1]])
        c = Parameter('c', dimensions=1, sw=1, values=[[1]])
        fir_x = Fir(W=a)(x.last())
        fir_y = Fir(W=b)(y.last())
        fir_z = Fir(W=c)(z.last())
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
        test.exportPythonModel(name=network_name)

        ## Load the exported model.py
        ## Import the python exported module
        #from results.exported_model import RecurrentModel
        module = importlib.import_module(result_path+'.'+network_name)
        RecurrentModel = getattr(module, 'RecurrentModel')
        model = RecurrentModel()

        # Create dummy input data
        dummy_input = {'x': torch.ones(5, 1, 1, 1), 'target': torch.ones(10, 1, 1, 1), 'y': torch.zeros(1,1,1), 'z':torch.zeros(1,1,1)}  # Adjust the shape as needed
        # Inference with imported model
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output['out'], [torch.tensor([[[2.]]]), torch.tensor([[[4.]]]), torch.tensor([[[6.]]]), torch.tensor([[[8.]]]), torch.tensor([[[10.]]])])

        if os.path.exists(test.getWorkspace()):
            shutil.rmtree(test.getWorkspace())

    def test_export_and_import_onnx_module(self):
        NeuObj.clearNames()
        result_path = 'results'
        test = Modely(visualizer=None, seed=42, workspace=result_path)
        x = Input('x')
        y = State('y')
        z = State('z')
        a = Parameter('a', dimensions=1, sw=1, values=[[1]])
        b = Parameter('b', dimensions=1, sw=1, values=[[1]])
        c = Parameter('c', dimensions=1, sw=1, values=[[1]])
        fir_x = Fir(W=a)(x.last())
        fir_y = Fir(W=b)(y.last())
        fir_z = Fir(W=c)(z.last())
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
        onnx_model_path = os.path.join(result_path, 'onnx', 'net.onnx')
        dummy_input = {'x':np.ones(shape=(3, 1, 1, 1)).astype(np.float32),
                       'y':np.ones(shape=(1, 1, 1)).astype(np.float32),
                       'z':np.ones(shape=(1, 1, 1)).astype(np.float32)}
        outputs = Modely().onnxInference(dummy_input,onnx_model_path)
        # Get the output
        expected_output = np.array([[[[3.]]], [[[5.]]], [[[7.]]]], dtype=np.float32)
        self.assertEqual(outputs[0].tolist(), expected_output.tolist())

        if os.path.exists(test.getWorkspace()):
            shutil.rmtree(test.getWorkspace())

    def test_export_and_import_onnx_module_easy(self):
        NeuObj.clearNames()
        result_path = 'results'
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
        output_nodely = test({'num_cycle':np.ones(shape=(10)).astype(np.float32).tolist(), 'x':np.ones(shape=(1)).astype(np.float32).tolist()})

        ## ONNX IMPORT
        onnx_model_path = os.path.join(result_path, 'onnx', 'net.onnx')
        outputs = Modely().onnxInference(inputs={'num_cycle':np.ones(shape=(10, 1, 1, 1)).astype(np.float32), 'x':np.ones(shape=(1, 1, 1)).astype(np.float32)}, path=onnx_model_path)
        self.assertEqual(output_nodely['out1'], outputs[0].squeeze().tolist())
        self.assertEqual(output_nodely['out2'], outputs[1].squeeze().tolist())

        if os.path.exists(test.getWorkspace()):
            shutil.rmtree(test.getWorkspace())

    # TODO modify adding the integral to the acc to compute the velocity
    def test_export_and_import_onnx_module_complex(self):
        # Create nnodely structure
        result_path = 'results'
        vehicle = Modely(visualizer=None, seed=2, workspace=result_path)

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
        breaking_force = -Relu(Fir(W_init = init_negexp, W_init_params={'size_index':0, 'first_value':0.002, 'lambda':3})(brake.sw(n)))
        gravity_force = Linear(W_init=init_constant, W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.last())
        fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
        local_model = LocalModel(input_function=lambda: Fir(W_init = init_negexp, W_init_params={'size_index':0, 'first_value':0.002, 'lambda':3}))
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
        onnx_model_path = os.path.join(result_path, 'onnx', 'net.onnx')
        outputs = Modely().onnxInference(sample, onnx_model_path)
        self.assertEqual(outputs[0][0][0].tolist(), model_inference['accelleration'])

        if os.path.exists(vehicle.getWorkspace()):
            shutil.rmtree(vehicle.getWorkspace())

    def test_export_python_module_recurrent(self):
        NeuObj.clearNames()
        result_path = 'results'
        network_name = 'net'
        test = Modely(visualizer=None, seed=42, workspace=result_path)
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
        out3 = Output('input1-out', input1.last())
        out4 = Output('input2-out', input2.last())
        out5 = Output('input3-out', input3.sw(4))
        out6 = Output('input4-out', input4.sw(4))
        out7 = Output('state1-out', state1.last())
        out8 = Output('state2-out', state2.last())

        test.addModel('model', [out1, out2, out3, out4, out5, out6, out7, out8])
        test.neuralizeModel()

        test.exportPythonModel(name=network_name)

        ## Load the exported model.py
        ## Import the python exported module
        #from results.net import RecurrentModel
        module = importlib.import_module(result_path+'.'+network_name)
        RecurrentModel = getattr(module, 'RecurrentModel')
        recurrent_model = RecurrentModel()

        ## Without Horizon and without batch
        recurrent_sample = {'input1': torch.rand(size=(1,1,1,1), dtype=torch.float32),
                            'input2': torch.rand(size=(1,1,1,3), dtype=torch.float32),
                            'input3': torch.rand(size=(1,1,4,1), dtype=torch.float32),
                            'input4': torch.rand(size=(1,1,4,3), dtype=torch.float32)}
        recurrent_sample['state1'] = torch.rand(size=(1,1,1), dtype=torch.float32)
        recurrent_sample['state2'] = torch.rand(size=(1,1,3), dtype=torch.float32)
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out1']).shape), [1,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out2']).shape), [1,1,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input1-out']).shape), [1,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input2-out']).shape), [1,1,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input3-out']).shape), [1,1,4,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input4-out']).shape), [1,1,4,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state1-out']).shape), [1,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state2-out']).shape), [1,1,1,3])

        ## With Horizon and without batch
        recurrent_sample = {'input1': torch.rand(size=(5,1,1,1), dtype=torch.float32),
                            'input2': torch.rand(size=(5,1,1,3), dtype=torch.float32),
                            'input3': torch.rand(size=(5,1,4,1), dtype=torch.float32),
                            'input4': torch.rand(size=(5,1,4,3), dtype=torch.float32)}
        recurrent_sample['state1'] = torch.rand(size=(1,1,1), dtype=torch.float32)
        recurrent_sample['state2'] = torch.rand(size=(1,1,3), dtype=torch.float32)
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out1']).shape), [5,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out2']).shape), [5,1,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input1-out']).shape), [5,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input2-out']).shape), [5,1,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input3-out']).shape), [5,1,4,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input4-out']).shape), [5,1,4,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state1-out']).shape), [5,1,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state2-out']).shape), [5,1,1,3])

        ## With Horizon and with batch
        recurrent_sample = {'input1': torch.rand(size=(5,2,1,1), dtype=torch.float32),
                            'input2': torch.rand(size=(5,2,1,3), dtype=torch.float32),
                            'input3': torch.rand(size=(5,2,4,1), dtype=torch.float32),
                            'input4': torch.rand(size=(5,2,4,3), dtype=torch.float32)}
        recurrent_sample['state1'] = torch.rand(size=(2,1,1), dtype=torch.float32)
        recurrent_sample['state2'] = torch.rand(size=(2,1,3), dtype=torch.float32)
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out1']).shape), [5,2,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['out2']).shape), [5,2,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input1-out']).shape), [5,2,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input2-out']).shape), [5,2,1,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input3-out']).shape), [5,2,4,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['input4-out']).shape), [5,2,4,3])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state1-out']).shape), [5,2,1,1])
        self.assertListEqual(list(torch.stack(recurrent_model(recurrent_sample)['state2-out']).shape), [5,2,1,3])

        if os.path.exists(test.getWorkspace()):
            shutil.rmtree(test.getWorkspace())

    def test_export_onnx_module_recurrent(self):
        result_path = 'results'
        test = Modely(visualizer=None, seed=42, workspace= result_path)
        onnx_model_path = os.path.join(result_path, 'onnx', 'net.onnx')
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

        if os.path.exists(test.getWorkspace()):
            shutil.rmtree(test.getWorkspace())

    def test_export_and_import_python_module_complex_recurrent(self):
        NeuObj.clearNames()
        # Create nnodely structure
        result_path = 'results'
        network_name = 'vehicle'
        vehicle = Modely(visualizer=None, seed=2, workspace=result_path)

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
        breaking_force = -Relu(Fir(W_init = init_negexp, W_init_params={'size_index':0, 'first_value':0.002, 'lambda':3})(brake.sw(n)))
        gravity_force = Linear(W_init=init_constant, W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.last())
        fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
        local_model = LocalModel(input_function=lambda: Fir(W_init = init_negexp, W_init_params={'size_index':0, 'first_value':0.002, 'lambda':3}))
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

        vehicle.exportPythonModel(name=network_name)

        loaded_vehicle = Modely(visualizer=None, workspace=vehicle.getWorkspace())
        loaded_vehicle.importPythonModel(name=network_name)
        model_import_inference = loaded_vehicle(sample, sampled=True, prediction_samples=3)
        self.assertEqual(model_inference['accelleration'], model_import_inference['accelleration'])

        ## Load the exported model.py
        ## Import the python exported module
        #from results.vehicle import RecurrentModel
        module = importlib.import_module(result_path+'.'+network_name)
        RecurrentModel = getattr(module, 'RecurrentModel')
        recurrent_model = RecurrentModel()

        sample = vehicle.getSamples('dataset', window=3)
        recurrent_sample = {key: torch.tensor(np.array(value), dtype=torch.float32).unsqueeze(1) for key, value in sample.items()}
        recurrent_sample['vel'] = torch.zeros(1,1,1)
        model_sample = {key: value for key, value in sample.items() if key != 'vel'}
        self.TestAlmostEqual([item.detach().item() for item in recurrent_model(recurrent_sample)['accelleration']], vehicle(model_sample, sampled=True, prediction_samples=3)['accelleration'])

        if os.path.exists(vehicle.getWorkspace()):
            shutil.rmtree(vehicle.getWorkspace())

    def test_export_and_import_onnx_module_complex_recurrent(self):
        NeuObj.clearNames()
        # Create nnodely structure
        result_path = 'results'
        network_name = 'vehicle'
        vehicle = Modely(visualizer=None, seed=42, workspace= result_path)

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
        breaking_force = -Relu(Fir(W_init = init_negexp, W_init_params={'size_index':0, 'first_value':0.002, 'lambda':3})(brake.sw(n)))
        gravity_force = Linear(W_init=init_constant, W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.last())
        fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
        local_model = LocalModel(input_function=lambda: Fir(W_init = init_negexp, W_init_params={'size_index':0, 'first_value':0.002, 'lambda':3}))
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
        vehicle.exportONNX(name=network_name)

        model_sample = vehicle.getSamples('dataset', window=1)
        model_inference = vehicle(model_sample, sampled=True, prediction_samples=1)

        ## ONNX IMPORT
        onnx_model_path = os.path.join(result_path, 'onnx', network_name+'.onnx')
        onnx_sample = {key: (np.expand_dims(value, axis=1).astype(np.float32) if key != 'vel' else value)  for key, value in model_sample.items()}
        outputs = Modely().onnxInference(onnx_sample, onnx_model_path)
        self.assertEqual(outputs[0][0], model_inference['accelleration'])

        model_sample = vehicle.getSamples('dataset', window=3)
        onnx_sample = {key: (np.expand_dims(value, axis=1).astype(np.float32) if key != 'vel' else np.expand_dims(np.array(value[0], dtype=np.float32), axis=0))  for key, value in model_sample.items()}
        model_inference = vehicle(model_sample, sampled=True, prediction_samples=3)
        outputs = Modely().onnxInference(onnx_sample, onnx_model_path)
        self.assertEqual(outputs[0].squeeze().tolist(), model_inference['accelleration'])

        if os.path.exists(vehicle.getWorkspace()):
            shutil.rmtree(vehicle.getWorkspace())

    def test_export_sw_on_stream_sw_complex(self):
        NeuObj.clearNames()
        Stream.resetCount()
        # Create nnodely structure
        result_path = 'results'
        network_name = 'swnet'
        input = Input('inin')

        sw_7 = input.sw(7)

        out61 = Output('out61', sw_7.sw(6))
        out62 = Output('out62', SamplePart(sw_7,1,7))
        test = Modely(workspace=result_path)
        test.addModel('out', [out61,out62])
        test.neuralizeModel()
        sample = [14, 1, 2, 3, 4, 5, 6]
        results = test({'inin':sample})

        onnx_model_path = os.path.join(result_path, 'onnx', network_name+'.onnx')
        test.exportONNX(inputs_order=['inin','SamplePart1_sw1'],outputs_order=['out61','out62'],name=network_name)
        outputs = Modely().onnxInference({'inin':np.array([[[[14],[1],[2],[3],[4],[5],[6]]]]).astype(np.float32),'SamplePart1_sw1':np.array([[[0],[0],[0],[0],[0],[0]]]).astype(np.float32)}, onnx_model_path)
        self.assertEqual(outputs[0].squeeze().tolist(), results['out61'][0])
        self.assertEqual(outputs[1].squeeze().tolist(), results['out62'][0])
        self.assertEqual(results['out61'][0], results['out62'][0])

        if os.path.exists(test.getWorkspace()):
            shutil.rmtree(test.getWorkspace())


