import sys, os, unittest

import numpy as np

from nnodely import *
from nnodely.relation import NeuObj, Stream
from nnodely.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 9 Tests
# This test file tests the json, in particular
# the dimensions that are propagated through the relations
# and the structure of the json itself

def myFun(K1,K2,p1,p2):
    import torch
    return p1*K1+p2*torch.sin(K2)

def myFun_out5(K1,p1):
    import torch
    return torch.stack([K1,K1,K1,K1,K1],dim=2).squeeze(-1)*p1


NeuObj.count = 0

class ModelyJsonTest(unittest.TestCase):
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:
            self.assertEqual(len(data1),len(data2))
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_input(self):
        input = Input('in1')
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'tw': [0,0], 'sw': [0, 0]}}, 'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {},'States': {}},input.json)

        #Discrete input removed
        #input = Input('in', values=[2,3,4])
        #self.assertEqual({'Inputs': {'in': {'dim': 1, 'discrete': [2,3,4], 'tw': [0,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {},'States': {}},input.json)

    def test_aritmetic(self):
        Stream.resetCount()
        NeuObj.clearNames()
        input = Input('in1')
        inlast = input.last()
        out = inlast+inlast
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'tw': [0,0], 'sw': [-1, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {},'States': {}, 'Relations': {'Add2': ['Add', ['SamplePart1', 'SamplePart1']],
               'SamplePart1': ['SamplePart', ['in1'], -1, [-1, 0]]}},out.json)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'tw': [-1,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {},'States': {}, 'Relations': {'Add7': ['Add', ['TimePart4', 'TimePart6']],
               'TimePart4': ['TimePart', ['in1'], -1, [-1, 0]],
               'TimePart6': ['TimePart', ['in1'], -1, [-1, 0]]}},out.json)
        out = input.tw(1) * input.tw(1)
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'tw': [-1,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'States': {}, 'Relations': {'Mul12': ['Mul', ['TimePart9', 'TimePart11']],
               'TimePart9': ['TimePart', ['in1'], -1, [-1, 0]],
               'TimePart11': ['TimePart', ['in1'], -1, [-1, 0]]}},out.json)
        out = input.tw(1) - input.tw(1)
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'tw': [-1,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'States': {}, 'Relations': {'Sub17': ['Sub', ['TimePart14', 'TimePart16']],
               'TimePart14': ['TimePart', ['in1'], -1, [-1, 0]],
               'TimePart16': ['TimePart', ['in1'], -1, [-1, 0]]}},out.json)
        input = Input('in2', dimensions = 5)
        inlast = input.last()
        out = inlast + inlast
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in2': {'dim': 5, 'tw': [0,0], 'sw': [-1, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'States': {}, 'Relations': {'Add20': ['Add', ['SamplePart19', 'SamplePart19']],
               'SamplePart19': ['SamplePart', ['in2'], -1, [-1, 0]]}},out.json)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in2': {'dim': 5, 'tw': [-1, 0], 'sw': [0, 0]}}, 'Functions': {}, 'Parameters': {},'Outputs': {}, 'States': {}, 'Relations': {'Add25': ['Add', ['TimePart22', 'TimePart24']],
               'TimePart22': ['TimePart', ['in2'], -1, [-1, 0]],
               'TimePart24': ['TimePart', ['in2'], -1, [-1, 0]]}}, out.json)
        out = input.tw([2,5]) + input.tw([3,6])
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in2': {'dim': 5, 'tw': [2, 6], 'sw': [0, 0]}}, 'Functions': {}, 'Parameters': {},'Outputs': {}, 'States': {}, 'Relations': {'Add30': ['Add', ['TimePart27', 'TimePart29']],
               'TimePart27': ['TimePart', ['in2'], -1, [2, 5]],
               'TimePart29': ['TimePart', ['in2'], -1, [3, 6]]}}, out.json)
        out = input.tw([-5,-2]) + input.tw([-6,-3])
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in2': {'dim': 5, 'tw': [-6, -2], 'sw': [0, 0]}}, 'Functions': {}, 'Parameters': {},'Outputs': {}, 'States': {}, 'Relations': {'Add35': ['Add', ['TimePart32', 'TimePart34']],
               'TimePart32': ['TimePart', ['in2'], -1, [-5, -2]],
               'TimePart34': ['TimePart', ['in2'], -1, [-6, -3]]}}, out.json)

    def test_scalar_input_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1').last()
        out = input+input
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = Fir(input)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = Fir(7)(input)
        self.assertEqual({'dim': 7,'sw': 1}, out.dim)
        out = Fuzzify(5, [-1,1])(input)
        self.assertEqual({'dim': 5,'sw': 1}, out.dim)
        out = ParamFun(myFun)(input)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = ParamFun(myFun_out5)(input)
        self.assertEqual({'dim': 5, 'sw': 1}, out.dim)
        with self.assertRaises(ValueError):
            out = Fir(Fir(7)(input))
        #
        with self.assertRaises(IndexError):
            out = Part(input,0,4)
        inpart = ParamFun(myFun_out5)(input)
        out = Part(inpart,0,4)
        self.assertEqual({'dim': 4, 'sw': 1}, out.dim)
        out = Part(inpart,0,1)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Part(inpart,1,3)
        self.assertEqual({'dim': 2, 'sw': 1}, out.dim)
        out = Select(inpart,0)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        with self.assertRaises(IndexError):
            out = Select(inpart,5)
        with self.assertRaises(IndexError):
            out = Select(inpart,-1)
        with self.assertRaises(KeyError):
            out = TimePart(inpart,-1,0)

    def test_scalar_input_tw_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1')
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)
        out = Fir(input.tw(1))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Fir(5)(input.tw(1))
        self.assertEqual({'dim': 5, 'sw': 1}, out.dim)
        out = Fuzzify(5, [0,5])(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = Fuzzify(5,range=[-1,5])(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = Fuzzify(centers=[-1,5])(input.tw(2))
        self.assertEqual({'dim': 2, 'tw': 2}, out.dim)
        out = ParamFun(myFun)(input.tw(1))
        self.assertEqual({'dim': 1, 'tw' : 1}, out.dim)
        out = ParamFun(myFun_out5)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = ParamFun(myFun_out5)(input.tw(2),input.tw(1))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        inpart = ParamFun(myFun_out5)(input.tw(2))
        out = Part(inpart,0,4)
        self.assertEqual({'dim': 4,'tw': 2}, out.dim)
        out = Part(inpart,0,1)
        self.assertEqual({'dim': 1,'tw': 2}, out.dim)
        out = Part(inpart,1,3)
        self.assertEqual({'dim': 2,'tw': 2}, out.dim)
        out = Select(inpart,0)
        self.assertEqual({'dim': 1,'tw': 2}, out.dim)
        with self.assertRaises(IndexError):
            out = Select(inpart,5)
        with self.assertRaises(IndexError):
            out = Select(inpart,-1)
        out = TimePart(inpart, 0,1)
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        #out = TimeSelect(inpart,0)
        #self.assertEqual({'dim': 5}, out.dim)
        #with self.assertRaises(ValueError):
        #   out = TimeSelect(inpart,-3)
        twinput = input.tw([-2,4])
        out = TimePart(twinput, 0, 1)
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)

    def test_scalar_input_tw2_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1')
        out = input.tw([-1,1])+input.tw([-2,0])
        self.assertEqual({'dim': 1, 'tw': 2}, out.dim)
        out = input.tw(1)+input.tw([-1,0])
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)
        out = Fir(input.tw(1) + input.tw([-1, 0]))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = input.tw([-1,0])+input.tw([-4,-3])+input.tw(1)
        self.assertEqual({'dim': 1,'tw': 1}, out.dim)
        with self.assertRaises(ValueError):
             out = input.tw([-2,0])-input.tw([-1,0])
        with self.assertRaises(ValueError):
             out = input.tw([-2,0])+input.tw([-1,0])

    def test_scalar_input_sw_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1')
        out = input.sw([-1,1])+input.sw([-2,0])
        self.assertEqual({'dim': 1, 'sw': 2}, out.dim)
        out = input.sw(1)+input.sw([-1,0])
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Fir(input.sw(1) + input.sw([-1, 0]))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = input.sw([-1,0])+input.sw([-4,-3])+input.sw(1)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        with self.assertRaises(ValueError):
            out = input.sw([-2,0])-input.sw([-1,0])
        with self.assertRaises(ValueError):
            out = input.sw([-2,0])+input.sw([-1,0])
        with self.assertRaises(ValueError):
            out = input.sw(1) + input.tw([-1, 0])
        with self.assertRaises(TypeError):
            out = input.sw(1.2)
        with self.assertRaises(TypeError):
            out = input.sw([-1.2,0.05])

    def test_vector_input_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1', dimensions = 5)
        self.assertEqual({'dim': 5}, input.dim)
        self.assertEqual({'dim': 5, 'tw' : 2}, input.tw(2).dim)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        out = Relu(input.tw(1))
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        with self.assertRaises(TypeError):
            Fir(7)(input)
        with self.assertRaises(TypeError):
            Fuzzify(7,[1,7])(input)
        out = ParamFun(myFun)(input.tw(1))
        self.assertEqual({'dim': 5, 'tw' : 1}, out.dim)
        out = ParamFun(myFun)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = ParamFun(myFun)(input.tw(2),input.tw(1))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)

    def test_parameter_and_linear(self):
        NeuObj.clearNames()
        input = Input('in1').last()
        W15 = Parameter('W15', dimensions=(1, 5))
        b15 = Parameter('b15', dimensions=5)
        input4 = Input('in4',dimensions=4).last()
        W45 = Parameter('W45', dimensions=(4, 5))
        b45 = Parameter('b45', dimensions=5)

        out = Linear(input) + Linear(input4)
        out3 = Linear(3)(input) + Linear(3)(input4)
        outW = Linear(W = W15)(input) + Linear(W = W45)(input4)
        outWb = Linear(W = W15,b = b15)(input) + Linear(W = W45, b = b45)(input4)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        self.assertEqual({'dim': 3, 'sw': 1}, out3.dim)
        self.assertEqual({'dim': 5, 'sw': 1}, outW.dim)
        self.assertEqual({'dim': 5, 'sw': 1}, outWb.dim)

        NeuObj.clearNames()
        input2 = Input('in1').sw([-1,1])
        W15 = Parameter('W15', dimensions=(1, 5))
        b15 = Parameter('b15', dimensions=5)
        input42 = Input('in4', dimensions=4).sw([-1,1])
        W45 = Parameter('W45', dimensions=(4, 5))
        b45 = Parameter('b45', dimensions=5)

        out = Linear(input2) + Linear(input42)
        out3 = Linear(3)(input2) + Linear(3)(input42)
        outW = Linear(W = W15)(input2) + Linear(W = W45)(input42)
        outWb = Linear(W = W15,b = b15)(input2) + Linear(W = W45, b = b45)(input42)
        self.assertEqual({'dim': 1, 'sw': 2}, out.dim)
        self.assertEqual({'dim': 3, 'sw': 2}, out3.dim)
        self.assertEqual({'dim': 5, 'sw': 2}, outW.dim)
        self.assertEqual({'dim': 5, 'sw': 2}, outWb.dim)

        with self.assertRaises(ValueError):
            Linear(input) + Linear(input42)
        with self.assertRaises(ValueError):
            Linear(3)(input2) + Linear(3)(input4)
        with self.assertRaises(ValueError):
            Linear(W = W15)(input) + Linear(W = W45)(input42)
        with self.assertRaises(ValueError):
            Linear(W = W15,b = b15)(input2) + Linear(W = W45, b = b45)(input4)

    def test_input_paramfun_param_const(self):
        NeuObj.clearNames()
        input2 = Input('in2')
        def fun_test(x,y,z,k):
            return x*y*z*k

        NeuObj.clearNames()
        out = ParamFun(fun_test)(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': 1,'sw': 1},'FParamFun0y': {'dim': 1,'sw': 1},'FParamFun0z': {'dim': 1,'sw': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test)(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': 1,'sw': 1}, 'FParamFun0z': {'dim': 1,'sw': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters=['t'])(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': 1,'sw': 1}, 't': {'dim': 1,'sw': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters=['t','r'])(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'r': {'dim': 1,'sw': 1}, 't': {'dim': 1,'sw': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters={'k':'t'})(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0z': {'dim': 1,'sw': 1}, 't': {'dim': 1,'sw': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_dimensions={'k':(1,2)})(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 2, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': [1,2],'sw': 1}, 'FParamFun0z': {'dim': 1,'sw': 1}}, out.json['Parameters'])

        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters_dimensions={'k':(1,2)},parameters=['r'])(input2.tw(0.01),input2.tw(0.01))
        with self.assertRaises(ValueError):
           ParamFun(fun_test,parameters_dimensions=[(1,2)],parameters={'z':'gg'})(input2.tw(0.01),input2.tw(0.01))
        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters_dimensions=[(1,2)],parameters=['pp'])(input2.tw(0.01))

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_dimensions={'k':(1,2)})(input2.tw(0.01),input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 2, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': [1,2],'sw': 1}}, out.json['Parameters'])

        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters_dimensions={'z':(1,2)})(input2.tw(0.01),input2.tw(0.01),input2.tw(0.01))

        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters={'z':'g'})(input2.tw(0.01),input2.tw(0.01),input2.tw(0.01))

        with self.assertRaises(ValueError):
            ParamFun(fun_test,constants={'z':'g'})(input2.tw(0.01),input2.tw(0.01),input2.tw(0.01))

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters=['pp'],constants=['el'])(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': 1,'sw': 1}, 'pp': {'dim': 1,'sw': 1}}, out.json['Parameters'])
        self.assertEqual({'el': {'dim': 1,'sw': 1}}, out.json['Constants'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters={'y':'pp'},constants={'k':'el'})(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0z': {'dim': 1,'sw': 1}, 'pp': {'dim': 1,'sw': 1}}, out.json['Parameters'])
        self.assertEqual({'el': {'dim': 1,'sw': 1}}, out.json['Constants'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters=['pp','oo'],constants={'k':'el'})(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'oo': {'dim': 1,'sw': 1}, 'pp': {'dim': 1,'sw': 1}}, out.json['Parameters'])
        self.assertEqual({'el': {'dim': 1,'sw': 1}}, out.json['Constants'])

        with self.assertRaises(NameError):
            ParamFun(fun_test,parameters=['pp','oo'],constants={'y':'el'})(input2.tw(0.01))

        with self.assertRaises(NameError):
            ParamFun(fun_test,parameters=['pp','oo'],constants={'z':'el'})(input2.tw(0.01))

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters=['pp'],constants=['ll','oo'])(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'pp': {'dim': 1,'sw': 1}}, out.json['Parameters'])
        self.assertEqual({'oo': {'dim': 1,'sw': 1}, 'll': {'dim': 1,'sw': 1}}, out.json['Constants'])

        NeuObj.clearNames()
        pp = Parameter('pp')
        ll = Constant('ll', values=[[1]])
        oo = Constant('oo', values=[[1]])
        out = ParamFun(fun_test,parameters=[pp],constants=[ll,oo])(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'pp': {'dim': 1,'sw': 1}}, out.json['Parameters'])
        self.assertEqual({'oo': {'dim': 1, 'sw': 1, 'values': [[1]]}, 'll': {'dim': 1, 'sw': 1, 'values': [[1]]}}, out.json['Constants'])

        out = ParamFun(fun_test,parameters={'z':pp},constants={'y':ll,'k':oo})(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'pp': {'dim': 1,'sw': 1}}, out.json['Parameters'])
        self.assertEqual(['ll', 'pp', 'oo'],out.json['Functions']['FParamFun4']['params_and_consts'])
        self.assertEqual({'oo': {'dim': 1, 'sw': 1, 'values': [[1]]}, 'll': {'dim': 1, 'sw': 1, 'values': [[1]]}}, out.json['Constants'])

        NeuObj.clearNames()
        Stream.resetCount()
        pp = Parameter('pp')
        ll = Constant('ll', values=[[1]])
        oo = Constant('oo', values=[[1]])
        out = ParamFun(fun_test)(input2.tw(0.01),ll,oo,pp)
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'pp': {'dim': 1,'sw': 1}}, out.json['Parameters'])
        self.assertEqual({'oo': {'dim': 1, 'sw': 1, 'values': [[1]]}, 'll': {'dim': 1, 'sw': 1, 'values': [[1]]}}, out.json['Constants'])
        self.assertEqual(['TimePart1', 'll', 'oo', 'pp'], out.json['Relations']['ParamFun2'][1])

    def test_check_multiple_streams_compatibility_paramfun(self):
        NeuObj.clearNames()
        log.setAllLevel(logging.WARNING)
        x = Input('x')
        F = Input('F')

        def myFun(p1, p2, k1, k2):
            import torch
            return k1 * torch.sin(p1) + k2 * torch.cos(p2)

        K1 = Parameter('k1', dimensions=1, sw=1, values=[[2.0]])
        K2 = Parameter('k2', dimensions=1, sw=1, values=[[3.0]])
        parfun = ParamFun(myFun, parameters=[K1, K2])

        rel1 = parfun(x.last(), F.last())
        rel2 = parfun(Tanh(F.sw(2)+F.sw([-2,-0])+F.sw([-3,-1])+F.sw([-4,-2])), Tanh(F.sw([0,2])))
        rel3 = parfun(Tanh(F.sw([-2,1])))
        rel4 = parfun(Tanh(F.sw([-2,1])), K1)
        rel5 = parfun(K1, Tanh(F.sw(1)))
        with self.assertRaises(TypeError):
            parfun(Fir(3)(parfun(x.tw(0.4), x.tw(0.4))))

        out1 = Output('out1', rel1)
        out2 = Output('out2', rel2)
        out3 = Output('out3', rel3)
        out4 = Output('out4', rel4)
        out5 = Output('out5', rel5)

        # m = MPLVisualizer(5)
        # m.showFunctions(list(example.model_def['Functions'].keys()), xlim=[[-5, 5], [-1, 1]])
        exampleA = Modely(seed=2)
        with self.assertRaises(TypeError):
            exampleA.addModel('model', [out1, out2, out3])
        exampleA.addModel('model_A', [out1, out2])
        with self.assertRaises(TypeError):
            exampleA.addModel('model_B', [out3])
        exampleA.addModel('model_A2', [out1, out2, out4, out5])
        exampleA.neuralizeModel(0.25)

        exampleB = Modely(seed=2)
        exampleB.addModel('model_B', [out3])
        exampleB.neuralizeModel(1)

        resultsA = exampleA({'x': [1, 3, 3]})
        self.TestAlmostEqual([4.682941913604736, 3.2822399139404297, 3.2822399139404297], resultsA['out1'])
        self.TestAlmostEqual([[3.0, 3.0],[3.0 , 3.0],[3.0 , 3.0]], resultsA['out2'])
        self.TestAlmostEqual([[-1.2484405040740967, -1.2484405040740967, -1.2484405040740967],
                             [-1.2484405040740967, -1.2484405040740967, -1.2484405040740967],
                             [-1.2484405040740967, -1.2484405040740967, -1.2484405040740967]], resultsA['out4'])
        self.TestAlmostEqual([4.818594932556152, 4.818594932556152, 4.818594932556152], resultsA['out5'])

        resultsB = exampleB({'F': [1, 3, 4]})
        self.TestAlmostEqual([[1.814424991607666, 2.2605631351470947, 2.267522096633911]], resultsB['out3'])

        log.setAllLevel(logging.CRITICAL)

    def test_check_multiple_streams_compatibility_linear(self):
        NeuObj.clearNames()
        log.setAllLevel(logging.WARNING)
        x = Input('x',dimensions=3)
        f = Input('f')

        lin = Linear()

        l1out = lin(x.last()) + Fir(lin(x.tw(2.0))) + Fir(lin(x.sw(3)))
        l2out = lin(f.last()) + Fir(lin(f.tw(2.0))) + Fir(lin(f.sw(3)))

        out1 = Output('out1', l1out)
        out2 = Output('out2', l2out)

        exampleA = Modely(visualizer=None, seed=2)
        with self.assertRaises(TypeError):
            exampleA.addModel('model', [out1, out2])
        exampleA.addModel('model_A', [out1])
        with self.assertRaises(TypeError):
            exampleA.addModel('model_B', [out2])
        exampleA.neuralizeModel(1)

        exampleB = Modely(visualizer=None, seed=2)
        exampleB.addModel('model_B', [out2])
        exampleB.neuralizeModel(1)

        resultsA = exampleA({'x': [[1, 3, 3], [1, 2, 1], [2, 3, 4]]})
        self.TestAlmostEqual([12.507442474365234], resultsA['out1'])

        resultsB = exampleB({'f': [1, 3, 3, 1, 2, 1]})
        self.TestAlmostEqual([6.585615158081055, 4.480303764343262, 4.106618881225586, 3.18161678314209], resultsB['out2'])

        log.setAllLevel(logging.CRITICAL)

    def test_check_multiple_streams_compatibility_fir(self):
        NeuObj.clearNames()
        log.setAllLevel(logging.WARNING)
        x = Input('x')

        fir = Fir()
        with self.assertRaises(TypeError):
            fir(x.last()) + fir(x.tw(2.0)) + fir(x.sw(3))

        out1 = Output('out1', fir(x.last()))
        out2 = Output('out2', fir(x.tw(2.0)))

        exampleA = Modely(visualizer=None, seed=2)
        with self.assertRaises(TypeError):
            exampleA.addModel('model', [out1, out2])
        exampleA.addModel('model_A', [out1])
        with self.assertRaises(TypeError):
            exampleA.addModel('model_B', [out2])
        exampleA.neuralizeModel(1)

        exampleB = Modely(visualizer=None, seed=2)
        exampleB.addModel('model_B', [out2])
        exampleB.neuralizeModel(1)

        resultsA = exampleA({'x': [1, 3]})
        self.TestAlmostEqual([0.6146950721740723, 1.8440852165222168], resultsA['out1'])

        resultsB = exampleB({'x': [1, 4, 5]})
        self.TestAlmostEqual([2.138746500015259, 4.363844871520996], resultsB['out2'])

        log.setAllLevel(logging.CRITICAL)

if __name__ == '__main__':
    unittest.main()
