import sys, os, unittest, torch, shutil
import numpy as np

from nnodely import *
from nnodely.relation import NeuObj
from nnodely.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 3 Tests
# Test of visualizers

class ModelyTestVisualizer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        NeuObj.clearNames()
        super(ModelyTestVisualizer, self).__init__(*args, **kwargs)

        self.x = x = Input('x')
        self.y = y = Input('y')
        self.z = z = Input('z')

        ## create the relations
        def myFun(K1, p1, p2):
            return K1 * p1 * p2

        K_x = Parameter('k_x', dimensions=1, tw=1, init='init_constant', init_params={'value': 1})
        K_y = Parameter('k_y', dimensions=1, tw=1)
        w = Parameter('w', dimensions=1, tw=1, init='init_constant', init_params={'value': 1})
        t = Parameter('t', dimensions=1, tw=1)
        c_v = Constant('c_v', tw=1, values=[[1], [2]])
        c = 5
        w_5 = Parameter('w_5', dimensions=1, tw=5)
        t_5 = Parameter('t_5', dimensions=1, tw=5)
        c_5 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        c_5_2 = Constant('c_5_2', tw=5, values=c_5)
        parfun_x = ParamFun(myFun, parameters_and_constants=[K_x,c_v])
        parfun_y = ParamFun(myFun, parameters_and_constants=[K_y])
        parfun_zz = ParamFun(myFun)
        fir_w = Fir(W=w_5)(x.tw(5))
        fir_t = Fir(W=t_5)(y.tw(5))
        time_part = TimePart(x.tw(5), i=1, j=3)
        sample_select = SampleSelect(x.sw(5), i=1)

        def fuzzyfun(x):
            return torch.sin(x)

        fuzzy = Fuzzify(output_dimension=4, range=[0, 4], functions=fuzzyfun)(x.tw(1))
        fuzzyTriang = Fuzzify(centers=[1, 2, 3, 7])(x.tw(1))

        self.out = Output('out', Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v)))
        self.out2 = Output('out2', Add(w, x.tw(1)) + Add(t, y.tw(1)) + Add(w, c))
        self.out3 = Output('out3', Add(fir_w, fir_t))
        self.out4 = Output('out4', Linear(output_dimension=1)(fuzzy+fuzzyTriang))
        self.out5 = Output('out5', Fir(time_part) + Fir(sample_select))
        self.out6 = Output('out6', LocalModel(output_function=Fir())(x.tw(1), fuzzy))
        self.out7 = Output('out7', parfun_zz(z.last()))
        self.out8 = Output('out8', Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v)) + Fir(parfun_zz(x.tw(5), t_5, c_5_2)))

    def test_export_textvisualizer(self):
        t = TextVisualizer(5)
        test = Modely(visualizer=t, seed=42, workspace='./results')
        test.addModel('modelA', self.out)
        test.addModel('modelB', [self.out2, self.out3, self.out4])
        test.addModel('modelC', [self.out4, self.out5, self.out6])
        test.addModel('modelD', self.out7)
        test.addMinimize('error1', self.x.last(), self.out)
        test.addMinimize('error2', self.y.last(), self.out3, loss_function='rmse')
        test.addMinimize('error3', self.z.last(), self.out6, loss_function='rmse')

        test.neuralizeModel(0.5)

        data_x = np.arange(0.0, 5, 0.1)
        data_y = np.arange(0.0, 5, 0.1)
        a, b = -1.0, 2.0
        dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
        params = {'num_of_epochs': 10, 'lr': 0.01}
        test.loadData(name='dataset', source=dataset)  # Create the datastest.trainModel(optimizer='SGD', training_params=params)  # Train the traced model
        t.showMinimize('error1')
        t.showMinimize('error2')
        t.showMinimize('error3')
        test.trainModel(optimizer='SGD', training_params=params)
        t.showWeights()
        test.trainModel(optimizer='SGD', training_params=params, validation_dataset=params)
        test.trainModel(optimizer='SGD', training_params=params, validation_dataset=params, closed_loop={'x':'out'}, prediction_samples=1)
        test.saveModel()
        test.loadModel()

        test.neuralizeModel(0.5)
        test.exportPythonModel()
        test.importPythonModel()
        test.exportReport()

    def test_export_mplvisualizer(self):
        m = MPLVisualizer(5)
        test = Modely(visualizer=m, seed=42)
        test.addModel('modelA', self.out)
        test.addModel('modelB', [self.out2, self.out3, self.out4])
        test.addModel('modelC', [self.out4, self.out5, self.out6])
        test.addModel('modelD', self.out7)
        test.addMinimize('error1', self.x.last(), self.out)
        test.addMinimize('error2', self.y.last(), self.out3, loss_function='rmse')
        test.addMinimize('error3', self.z.last(), self.out6, loss_function='rmse')

        test.neuralizeModel(0.5)

        data_x = np.arange(0.0, 10, 0.1)
        data_y = np.arange(0.0, 10, 0.1)
        a, b = -1.0, 2.0
        dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
        params = {'num_of_epochs': 10, 'lr': 0.01}
        test.loadData(name='dataset', source=dataset)  # Create the dataset
        test.trainModel(optimizer='SGD', training_params=params)  # Train the traced model
        test.trainModel(optimizer='SGD', training_params=params)
        m.closeResult()
        m.closeTraining()
        list_of_functions = list(test.model_def['Functions'].keys())
        with self.assertRaises(ValueError):
            m.showFunctions(list_of_functions[1])
        m.closeFunctions()
        m.showFunctions(list_of_functions[0])
        m.showFunctions(list_of_functions[4])
        m.showFunctions(list_of_functions[3])
        m.closeFunctions()

    def test_export_mplvisualizer2(self):
        x = Input('x')
        F = Input('F')
        def myFun(K1, K2, p1, p2):
            import torch
            return p1 * K1 + p2 * torch.sin(K2)

        parfun = ParamFun(myFun)
        out = Output('fun', parfun(x.last(), F.last()))
        m = MPLVisualizer()
        example = Modely(visualizer=m)
        example.addModel('out', out)
        example.neuralizeModel()
        m.showFunctions(list(example.model_def['Functions'].keys()), xlim=[[-5, 5], [-1, 1]])
        m.closeFunctions()

    # def test_export_mplnotebookvisualizer(self):
    #     m = MPLNotebookVisualizer(5)
    #     test = Modely(visualizer=m, seed=42)
    #     test.addModel('modelA', self.out)
    #     test.addModel('modelB', [self.out2, self.out3, self.out4])
    #     test.addModel('modelC', [self.out4, self.out5, self.out6])
    #     test.addMinimize('error1', self.x.last(), self.out)
    #     test.addMinimize('error2', self.y.last(), self.out3, loss_function='rmse')
    #     test.addMinimize('error3', self.z.last(), self.out6, loss_function='rmse')
    #
    #     test.neuralizeModel(0.5)
    #
    #     data_x = np.arange(0.0, 10, 0.1)
    #     data_y = np.arange(0.0, 10, 0.1)
    #     a, b = -1.0, 2.0
    #     dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
    #     params = {'num_of_epochs': 1, 'lr': 0.01}
    #     test.loadData(name='dataset', source=dataset)  # Create the dataset
    #     test.trainModel(optimizer='SGD', training_params=params)  # Train the traced model
    #     m.showFunctions(list(test.model_def['Functions'].keys()), xlim=[[-5, 5], [-1, 1]])
    #     m.closeFunctions()