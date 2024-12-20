import copy, sys, os, torch
import numpy as np
# append a new directory to sys.path
sys.path.append(os.getcwd())

from nnodely import *

example = 3

if example == 1:
    print("-----------------------------------EXAMPLE 1------------------------------------")
    result_path = './results'
    test = Modely(seed=42, workspace=result_path)

    x = Input('x')
    y = State('y')

    a = Parameter('a', dimensions=1, sw=1)
    b = Parameter('b', dimensions=1, sw=1)

    fir_x = Fir(parameter=a)(x.last())
    fir_y = Fir(parameter=b)(y.last())
    sum_rel = fir_x+fir_y

    sum_rel.closedLoop(y)

    out = Output('out', sum_rel)

    test.addModel('model', out)
    #test.addClosedLoop(out, y)
    test.neuralizeModel(0.5)

    # Export the all models in onnx format
    test.exportONNX(['x','y'],['out']) # Export the onnx model

elif example == 2:
    print("-----------------------------------EXAMPLE 2------------------------------------")
    result_path = './results'
    test = Modely(seed=42, workspace=result_path)

    x = Input('x')
    y = State('y')

    a = Parameter('a', dimensions=1, sw=1)
    b = Parameter('b', dimensions=1, sw=1)

    fir_x = Fir(parameter=a)(x.last())
    fir_y = Fir(parameter=b)(y.last())
    fir_x.connect(y)
    out = Output('out', fir_x+fir_y)

    test.addModel('model', out)
    #test.addConnect(fir_x, y)
    test.neuralizeModel(0.5)

    # Export the all models in onnx format
    test.exportONNX(['x','y'],['out']) # Export the onnx model

elif example == 3:
    print("-----------------------------------EXAMPLE 3------------------------------------")
    result_path = './results'
    test = Modely(seed=42, workspace=result_path, visualizer=TextVisualizer(verbose=1))

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
    #out_b = Output('out_b', fir_y)

    test.addModel('model', out)
    #test.addModel('model_b', out_b)
    #test.addConnect(fir_x, y)
    test.addMinimize('error', target.last(), out)
    test.neuralizeModel(0.5)
    test.loadData(name='test_dataset', source=dataset)

    ## Train
    test.trainModel(optimizer='SGD', training_params={'num_of_epochs': 1, 'lr': 0.0001, 'train_batch_size': 1}, splits=[100,0,0], prediction_samples=10)

    ## Inference
    sample = {'x':[1], 'y':[2], 'z':[3], 'target':[18]}
    print(test(sample))
    print(test.model.all_parameters['a'])
    print(test.model.all_parameters['b'])
    print(test.model.all_parameters['c'])

    # Export the model
    test.exportPythonModel()

    # Import the model
    test.importPythonModel(name='net')

    ## Inference with imported model
    print(test(sample))
    print(test.model.all_parameters['a'])
    print(test.model.all_parameters['b'])
    print(test.model.all_parameters['c'])

    ## Train imported model
    test.trainModel(optimizer='SGD', training_params={'num_of_epochs': 1, 'lr': 0.0001, 'train_batch_size': 1}, splits=[100,0,0], prediction_samples=10)

    ## Inference
    sample = {'x':[1], 'y':[2], 'z':[3], 'target':[18]}
    print(test(sample))
    print(test.model.all_parameters['a'])
    print(test.model.all_parameters['b'])
    print(test.model.all_parameters['c'])

    ## Export in ONNX format
    test.exportONNX(['x','y','z'],['out']) # Export the onnx model

elif example == 4:
    print("-----------------------------------EXAMPLE 4------------------------------------")
    result_path = './results'
    test = Modely(seed=42, workspace=result_path, visualizer=TextVisualizer(verbose=0))

    x = Input('x')
    y = Input('y')
    z = Input('z')
    w = Input('w')
    target = Input('target')

    a = Parameter('a', dimensions=1, sw=1, values=[[1]])
    b = Parameter('b', dimensions=1, sw=1, values=[[1]])
    c = Parameter('c', dimensions=1, sw=1, values=[[1]])

    fir_x = Fir(parameter=a)(x.last())
    fir_y = Fir(parameter=b)(y.last())
    fir_z = Fir(parameter=c)(z.last())

    data_x, data_y, data_z = np.random.rand(100), np.random.rand(100), np.random.rand(100)
    dataset = {'x':data_x, 'y':data_y, 'z':data_z, 'target':3*data_x + 3*data_y + 3*data_z}

    sum_rel = fir_x + fir_y + fir_z

    out = Output('out', sum_rel)
    #out_b = Output('out_b', fir_y)

    test.addModel('model', out)
    #test.addModel('model_b', out_b)
    #test.addConnect(fir_x, y)
    test.addMinimize('error', target.last(), out)
    test.neuralizeModel(0.5)
    test.loadData(name='test_dataset', source=dataset)

    ## Train
    test.trainModel(optimizer='SGD', training_params={'num_of_epochs': 1, 'lr': 0.0001, 'train_batch_size': 1}, splits=[100,0,0], prediction_samples=10)

    ## Inference
    sample = {'x':[1], 'y':[2], 'z':[3], 'target':[18]}
    print(test(sample))
    print(test.model.all_parameters['a'])
    print(test.model.all_parameters['b'])
    print(test.model.all_parameters['c'])

    # Export the model
    test.exportPythonModel()

    # Import the model
    test.importPythonModel(name='net')

    ## Inference with imported model
    print(test(sample))
    print(test.model.all_parameters['a'])
    print(test.model.all_parameters['b'])
    print(test.model.all_parameters['c'])

    ## Train imported model
    test.trainModel(optimizer='SGD', training_params={'num_of_epochs': 1, 'lr': 0.0001, 'train_batch_size': 1},closed_loop={'y':'out'}, splits=[100,0,0], prediction_samples=10)

    ## Inference
    sample = {'x':[1], 'y':[2], 'z':[3], 'target':[18]}
    print(test(sample))
    print(test.model.all_parameters['a'])
    print(test.model.all_parameters['b'])
    print(test.model.all_parameters['c'])

    ## Export in ONNX format
    #test.exportONNX(['x','y','z'],['out']) # Export the onnx model

