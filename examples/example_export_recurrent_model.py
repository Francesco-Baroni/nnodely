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
    test = Modely(seed=42, workspace=result_path)

    x = Input('x')
    y = State('y')
    z = State('z')

    a = Parameter('a', dimensions=1, sw=1)
    b = Parameter('b', dimensions=1, sw=1)
    c = Parameter('c', dimensions=1, sw=1)

    fir_x = Fir(parameter=a)(x.last())
    fir_y = Fir(parameter=b)(y.last())
    fir_z = Fir(parameter=c)(z.last())

    fir_x.connect(y)

    sum_rel = fir_x + fir_y + fir_z

    sum_rel.closedLoop(z)

    out = Output('out', sum_rel)

    test.addModel('model', out)
    #test.addConnect(fir_x, y)
    test.neuralizeModel(0.5)

    # Export the all models in onnx format
    test.exportONNX(['x','y','z'],['out']) # Export the onnx model
