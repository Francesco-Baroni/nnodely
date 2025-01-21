import sys, os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import torch
from nnodely import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
def func1(K1):
    return torch.sin(K1)

def func2(K2):
    return torch.cos(K2)

parfun1 = ParamFun(func1)
parfun2 = ParamFun(func2)
equation_learner = EquationLearner([func1, func2])

out = Output('out',equation_learner(x.last()))
example = Modely(visualizer=MPLVisualizer())
example.addModel('eqlearner',out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
print(example({'x':[1,2],'F':[1,2]}))
example.visualizer.showFunctions(list(example.model_def['Functions'].keys()),xlim=[[-5,5],[-1,1]])