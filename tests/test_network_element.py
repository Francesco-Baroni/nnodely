import unittest, sys, os, torch

from nnodely import *
from nnodely.relation import Stream
from nnodely import relation
relation.CHECK_NAMES = False

from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 11 Tests
# This file tests the dimensions and the of the element created in the pytorch environment

class ModelyNetworkBuildingTest(unittest.TestCase):

    def test_network_building_very_simple(self):

        input1 = Input('in1').last()
        rel1 = Fir(input1)
        fun = Output('out', rel1)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[1, 1], [1, 1]]
        for ind, (key, value) in enumerate({k: v for k, v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))
      
    def test_network_building_simple(self):
        Stream.reset_count()
        input1 = Input('in1')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output('out',rel1+rel2)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)
        
        list_of_dimensions = {'Fir3':[5,1],'Fir6':[1,1]}
        for key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw(self):
        Stream.reset_count()
        input1 = Input('in1')
        input2 = Input('in2')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        fun = Output('out',rel1+rel2+rel3+rel4)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)
        
        list_of_dimensions = {'Fir4':[5,1], 'Fir7':[1,1], 'Fir10':[5,1], 'Fir13':[4,1]}
        for key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))
    
    def test_network_building_tw2(self):
        Stream.reset_count()
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        rel5 = Fir(input2.tw([-0.03,0.03]))
        rel6 = Fir(input2.tw([-0.03, 0]))
        rel7 = Fir(input2.tw(0.03))
        fun = Output('out',rel3+rel4+rel5+rel6+rel7)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 8) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 8} , test.input_n_samples)
        
        list_of_dimensions = {'Fir3':[5,1], 'Fir6':[4,1], 'Fir9':[6,1], 'Fir12':[3,1], 'Fir15':[3,1]}
        for  key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw3(self):
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.01,0.03]))
        rel5 = Fir(input2.tw([-0.04,0.01]))
        fun = Output('out',rel3+rel4+rel5)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[5,1], [4,1], [5,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_tw_with_offest(self):
        Stream.reset_count()
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.04,0.02]))
        rel5 = Fir(input2.tw([-0.04,0.02],offset=0))
        rel6 = Fir(input2.tw([-0.04,0.02],offset=0.01))
        fun = Output('out',rel3+rel4+rel5+rel6)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = {'Fir3':[5,1], 'Fir6':[6,1], 'Fir9':[6,1], 'Fir12':[6,1]}
        for key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw_negative(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([-0.04,-0.01]))
        rel2 = Fir(input2.tw([-0.06,-0.03]))
        fun = Output('out',rel1+rel2)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_tw_positive(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([0.01,0.04]))
        rel2 = Fir(input2.tw([0.03,0.06]))
        fun = Output('out',rel1+rel2)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_sw_with_offset(self):
        Stream.reset_count()
        input2 = Input('in2')
        rel3 = Fir(input2.sw(5))
        rel4 = Fir(input2.sw([-4,2]))
        rel5 = Fir(input2.sw([-4,2],offset=0))
        rel6 = Fir(input2.sw([-4,2],offset=1))
        fun = Output('out',rel3+rel4+rel5+rel6)

        test = Modely(visualizer=None, seed=1)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = {'Fir3':[5,1], 'Fir6':[6,1], 'Fir9':[6,1], 'Fir12':[6,1]}
        for key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_sw_and_tw(self):
        input2 = Input('in2')
        with self.assertRaises(ValueError):
            input2.sw(5)+input2.tw(0.05)

        rel1 = Fir(input2.sw([-4,2]))+Fir(input2.tw([-0.01,0]))
        fun = Output('out',rel1)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[6,1], [1,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_linear(self):
        torch.manual_seed(1)
        input = Input('in')
        rel1 = Linear(input.sw([-4,2]))
        rel2 = Linear(5)(input.sw([-1, 2]))
        fun1 = Output('out1',rel1)
        fun2 = Output('out2', rel2)

        input5 = Input('in5', dimensions=3)
        rel15 = Linear(input5.sw([-4,2]))
        rel25 = Linear(5)(input5.last())
        fun15 = Output('out51',rel15)
        fun25 = Output('out52', rel25)

        test = Modely(visualizer=None)
        test.addModel('fun',[fun1,fun2,fun15,fun25])
        test.neuralizeModel(0.01)

        list_of_dimensions = [[1,1,1],[1,1,5],[1,3,1],[1,3,5]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Linear' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_sigmoid_function(self):
        torch.manual_seed(1)
        input = Input('in')
        sigma_rel = Sigma(input.last())
        sigma_rel_2 = Sigma(input.sw(2))

        input5 = Input('in5', dimensions=5)
        sigma_rel_5 = Sigma(input5.last())

        out1 = Output('out1', sigma_rel)
        out2 = Output('out2', sigma_rel_5)
        out3 = Output('out3', sigma_rel_2)

        test = Modely(visualizer=None)
        test.addModel('model',[out1,out2,out3])
        test.neuralizeModel(0.01)

        result = test(inputs={'in':[[3.0],[-2.0]], 'in5':[[4.0,1.0,0.0,-6.0,2.0]]})
        self.assertEqual([0.11920291930437088], result['out1'])
        self.assertEqual([[[0.9820137619972229, 0.7310585975646973, 0.5, 0.0024726230185478926, 0.8807970285415649]]], result['out2'])
        self.assertEqual([[0.9525741338729858, 0.11920291930437088]], result['out3'])

    def test_sech_cosh_function(self):
        torch.manual_seed(1)
        input = Input('in')
        sech_rel = Sech(input.last())
        sech_rel_2 = Sech(input.sw(2))
        cosh_rel = Cosh(input.last())
        cosh_rel_2 = Cosh(input.sw(2))

        input5 = Input('in5', dimensions=5)
        sech_rel_5 = Sech(input5.last())
        cosh_rel_5 = Cosh(input5.last())

        out1 = Output('sech_out_1', sech_rel)
        out2 = Output('sech_out_2', sech_rel_2)
        out3 = Output('sech_out_3', sech_rel_5)
        out4 = Output('cosh_out_1', cosh_rel)
        out5 = Output('cosh_out_2', cosh_rel_2)
        out6 = Output('cosh_out_3', cosh_rel_5)

        test = Modely(visualizer=None)
        test.addModel('model',[out1,out2,out3,out4,out5,out6])
        test.neuralizeModel(0.01)

        result = test(inputs={'in':[[3.0],[-2.0]], 'in5':[[4.0,1.0,0.0,-6.0,2.0]]})
        self.assertEqual([0.2658022344112396], result['sech_out_1'])
        self.assertEqual([[0.0993279218673706, 0.2658022344112396]], result['sech_out_2'])
        self.assertEqual([[[0.03661899268627167, 0.6480542421340942, 1.0, 0.004957473836839199, 0.2658022344112396]]], result['sech_out_3'])
        self.assertEqual([3.762195587158203], result['cosh_out_1'])
        self.assertEqual([[10.067662239074707, 3.762195587158203]], result['cosh_out_2'])
        self.assertEqual([[[27.3082332611084, 1.5430806875228882, 1.0, 201.71563720703125, 3.762195587158203]]], result['cosh_out_3'])

if __name__ == '__main__':
    unittest.main()