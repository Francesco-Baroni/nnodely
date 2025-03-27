import unittest, os, sys, torch
import numpy as np

from nnodely import *
from nnodely.relation import NeuObj
from nnodely.logger import logging, nnLogger

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
        test = Modely(visualizer=None, seed=1, device='gpu')
        test.addModel('out',out)
        test.neuralizeModel(0.01)
        results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],'in2': [[5]]})
        print(results)