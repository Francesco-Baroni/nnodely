import copy, sys, os, torch
import numpy as np
# append a new directory to sys.path
sys.path.append(os.getcwd())

from nnodely import *

example = 6

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
    #test.exportPythonModel()

    # Import the model
    #test.importPythonModel(name='net')

    ## Inference with imported model
    #print(test(sample))
    #print(test.model.all_parameters['a'])
    #print(test.model.all_parameters['b'])
    #print(test.model.all_parameters['c'])

    ## Train imported model
    #test.trainModel(optimizer='SGD', training_params={'num_of_epochs': 1, 'lr': 0.0001, 'train_batch_size': 1}, splits=[100,0,0], prediction_samples=10)

    ## Inference
    #sample = {'x':[1], 'y':[2], 'z':[3], 'target':[18]}
    #print(test(sample))
    #print(test.model.all_parameters['a'])
    #print(test.model.all_parameters['b'])
    #print(test.model.all_parameters['c'])

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

elif example == 5:
    print("-----------------------------------EXAMPLE Vehicle (Not Recurrent)------------------------------------")
    # Create nnodely structure
    vehicle = nnodely(visualizer=MPLVisualizer(),seed=2, workspace=os.path.join(os.getcwd(), 'results')) #MPLVisualizer()

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

    ## Export the Onnx Model
    vehicle.exportONNX(['vel','brk','gear','trq','alt'],['accelleration'])

elif example == 6:
    print("-----------------------------------EXAMPLE Vehicle (Recurrent)------------------------------------")
    # Create nnodely structure
    vehicle = nnodely(visualizer=MPLVisualizer(),seed=2, workspace=os.path.join(os.getcwd(), 'results')) #MPLVisualizer()

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

    data = {'vel':np.random.rand(1,1), 'brk':np.random.rand(25,1), 'gear':np.random.rand(1,1), 'trq':np.random.rand(25,1), 'alt':np.random.rand(1,21), 'acc':np.random.rand(1,1)}
    inference = vehicle(data)

    ## Export the Onnx Model
    vehicle.exportONNX(['brk','gear','trq','alt','vel'],['accelleration'])

    data = {'vel':np.random.rand(1,1,1).astype(np.float32), 'brk':np.random.rand(2,25,1).astype(np.float32), 'gear':np.random.rand(2,1,1).astype(np.float32), 'trq':np.random.rand(2,25,1).astype(np.float32), 'alt':np.random.rand(2,1,21).astype(np.float32), 'acc':np.random.rand(2,1,1).astype(np.float32)}
    print(Modely().onnxInference(data,'results/onnx/net.onnx'))