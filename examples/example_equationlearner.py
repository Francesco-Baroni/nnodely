import sys, os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import torch
from nnodely import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
## Create an equation learner block with pure nnodely functions
#W = Parameter(name='W', dimensions=[1,3], values=[[[1.0,1.0,1.0]]])
linear_layer = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)(x.last())
cos_layer = Cos(Select(linear_layer,0))
sin_layer = Sin(Select(linear_layer,1))
tan_layer = Tan(Select(linear_layer,2))
concatenate_layer = Concatenate(Concatenate(tan_layer,sin_layer),cos_layer)

#equation_learner = EquationLearner(functions=[Sin, Cos, Tan])
#out = Output('out',equation_learner(x.last()))
out_linear = Output('out_linear',linear_layer)
out_cos = Output('out_cos',cos_layer)
out_sin = Output('out_sin',sin_layer)
out_tan = Output('out_tan',tan_layer)
out_concatenate = Output('out_concatenate',concatenate_layer)
example = Modely(visualizer=MPLVisualizer())
#example.addModel('eqlearner',out)
example.addModel('model',[out_linear,out_cos,out_sin,out_tan,out_concatenate])
example.neuralizeModel()
print(example({'x':[1]}))

print("------------------------EXAMPLE 2------------------------")
## Create an equation learner block similar to Example 1 but using EquationLearner function
equation_learner = EquationLearner(functions=[Tan, Sin, Cos])
out = Output('out',equation_learner(x.last()))
example = Modely(visualizer=MPLVisualizer())
example.addModel('model',[out])
example.neuralizeModel()
print(example({'x':[1]}))

# print("------------------------EXAMPLE 3------------------------")
# ## Create an equation learner block with parametric functions
# def func1(K1):
#     return torch.sin(K1)

# def func2(K2):
#     return torch.cos(K2)

# parfun1 = ParamFun(func1)
# parfun2 = ParamFun(func2)
# equation_learner = EquationLearner([func1, func2])

# out = Output('out',equation_learner(x.last()))
# example = Modely(visualizer=MPLVisualizer())
# example.addModel('eqlearner',out)
# example.neuralizeModel()
# print(example({'x':[1],'F':[1]}))
# print(example({'x':[1,2],'F':[1,2]}))