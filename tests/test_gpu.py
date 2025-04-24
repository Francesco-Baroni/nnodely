import unittest, os, sys, torch
import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

#data_folder = os.path.join(os.path.dirname(__file__), 'data/')

class ModelyGpuTest(unittest.TestCase):
    def test_simple_gpu_inference(self):
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y')
        target = Input('target')
        out_fun = Fir(x.tw(0.1)) + Fir(y.last())
        out = Output('out', out_fun)
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',out)
        test.addMinimize('error', out, target.last())
        test.neuralizeModel(0.01, device='cpu')
        print(test._device)

        def fun(x, y, a, b):
            return a*x + a*y + b
        x_data = np.linspace(1, 20, 20)
        y_data = np.linspace(1, 20, 20)
        target_data = fun(x_data, y_data, 2, 3)
        dataset = {'x': x_data, 'y': y_data, 'target': target_data}
        test.loadData('dataset', dataset)

        test.trainModel(lr=0.01, train_dataset='dataset', num_of_epochs=1, train_batch_size=1, shuffle_data=False)

        test.neuralizeModel(0.01, device='gpu')
        print(test._device)
        test.trainModel(lr=0.01, train_dataset='dataset', num_of_epochs=1, train_batch_size=1, shuffle_data=False)
        #results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],'in2': [[5]]})
        #print(results)

    def test_gpu_export(self):
        NeuObj.clearNames()
        x = Input('x')
        y = State('y')
        c1 = Constant('c1', values=5.0)
        c3 = Constant('c3', values=5.0)
        rel2 = Linear(W=Parameter('W2', values=[[2.0]]), b=False)(y.last())
        rel4 = Linear(W=Parameter('W4', values=[[4.0]]), b=False)(y.last())
        rel2.closedLoop(y)
        def fun2(x, a):
            return x * a
        def fun4(x, b):
            return x + b
        out1 = Output('out1', c1 + Linear(W=Parameter('W1', values=[[1.0]]), b=False)(x.last()))
        out2 = Output('out2', rel2 + ParamFun(fun2, parameters_and_constants=[c1])(y.last()))
        out3 = Output('out3', c3 + Linear(W=Parameter('W3', values=[[3.0]]), b=False)(x.last()))
        out4 = Output('out4', rel4 + ParamFun(fun4, parameters_and_constants=[c3])(y.last()))

        result_path = './results'
        nn = Modely(visualizer=TextVisualizer(), workspace=result_path)
        nn.addModel('model', [out1, out2, out3, out4])
        nn.neuralizeModel(2.0)
        nn.exportPythonModel()

    def test_simple_gpu_inference_recurrent(self):
        NeuObj.clearNames()
        x = Input('x')
        y = State('y')
        target = Input('target')
        out_fun = Fir(x.tw(0.1)) + Fir(y.last())
        out_fun.closedLoop(y)
        out = Output('out', out_fun)
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',out)
        test.addMinimize('error', out, target.last())
        test.neuralizeModel(0.01, device='cpu')
        print(test._device)

        def fun(x, y, a, b):
            return a*x + a*y + b
        x_data = np.linspace(1, 20, 20)
        y_data = np.linspace(1, 20, 20)
        target_data = fun(x_data, y_data, 2, 3)
        dataset = {'x': x_data, 'y': y_data, 'target': target_data}
        test.loadData('dataset', dataset)

        test.trainModel(lr=0.01, train_dataset='dataset', num_of_epochs=1, train_batch_size=1, shuffle_data=False, prediction_samples=3)

        test.neuralizeModel(0.01, device='gpu')
        print(test._device)
        test.trainModel(lr=0.01, train_dataset='dataset', num_of_epochs=1, train_batch_size=1, shuffle_data=False, prediction_samples=3)
        #results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],'in2': [[5]]})
        #print(results)