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
        in1 = Input('in1')
        in2 = Input('in2')
        out_fun = Fir(in1.tw(0.1)) + Fir(in2.last())
        out = Output('out', out_fun)
        test = Modely(visualizer=None, seed=1, device='cpu')
        test.addModel('out',out)
        test.neuralizeModel(0.01)

        def fun(x, a, b):
            return a*x + b
        x_data = np.linspace(1, 20, 20)
        target_data = fun(x_data, 2, 3)
        dataset = {'x': x_data, 'target': target_data}
        test.loadData('dataset', dataset)
        test.trainModel(lr=0.01, train_dataset='dataset', num_of_epochs=1, train_batch_size=1, shuffle_data=False)
        self.assertEqual(test.device, 'cuda')
        results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],'in2': [[5]]})
        print(results)