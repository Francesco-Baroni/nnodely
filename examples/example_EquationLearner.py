import sys, os

from nnodely.relation import NeuObj

# append a new directory to sys.path
sys.path.append(os.getcwd())

import torch
from nnodely import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
## Create an equation learner block with pure nnodely functions
linear_layer = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)(x.last())
cos_layer = Cos(Select(linear_layer,0))
sin_layer = Sin(Select(linear_layer,1))
tan_layer = Tan(Select(linear_layer,2))
concatenate_layer = Concatenate(Concatenate(tan_layer,sin_layer),cos_layer)

out_linear = Output('out_linear',linear_layer)
out_cos = Output('out_cos',cos_layer)
out_sin = Output('out_sin',sin_layer)
out_tan = Output('out_tan',tan_layer)
out_concatenate = Output('out_concatenate',concatenate_layer)
example = Modely(visualizer=None)
example.addModel('model',[out_linear,out_cos,out_sin,out_tan,out_concatenate])
example.neuralizeModel()
print(example({'x':[1]}))

print("------------------------EXAMPLE 2------------------------")
## Create an equation learner block similar to Example 1 but using EquationLearner function
equation_learner = EquationLearner(functions=[Tan, Sin, Cos])
equation_learner_2 = EquationLearner(functions=[Tan, Sin, Cos])
out = Output('out',equation_learner(x.last()))
out2 = Output('out2',equation_learner_2(inputs=(x.last(),F.last())))
example = Modely(visualizer=None)
example.addModel('model',[out,out2])
example.neuralizeModel()
print(example({'x':[1], 'F':[2]}))

print("------------------------EXAMPLE 3------------------------")
NeuObj.clearNames(['out','out2'])
## Create an equation learner block similar to Example 1 but using EquationLearner function and passing a the linear layer as input
linear_layer_1 = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)
linear_layer_2 = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)
equation_learner = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer_1)
equation_learner_2 = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer_2)
out = Output('out',equation_learner(x.last()))
out2 = Output('out2',equation_learner_2(inputs=(x.last(),F.last())))
example = Modely(visualizer=None)
example.addModel('model',[out, out2])
example.neuralizeModel()
print(example({'x':[1], 'F':[2]}))

print("------------------------EXAMPLE 4------------------------")
NeuObj.clearNames(['out','out2'])
## Create an equation learner block similar to Example 1 but using EquationLearner function and passing a the linear layer as input and a linear layer as output
linear_layer_in = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)
linear_layer_out = Linear(output_dimension=1, W_init=init_constant, W_init_params={'value':1}, b_init=init_constant, b_init_params={'value':0}, b=False)
equation_learner = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer_in, linear_out=linear_layer_out)
out = Output('out',equation_learner(x.last()))
example = Modely(visualizer=None)
example.addModel('model',[out])
example.neuralizeModel()
print(example({'x':[1]}))

print("------------------------EXAMPLE 5------------------------")
NeuObj.clearNames(['out','out2'])
## Create an equation learner block with functions that take 2 parameters (add, sub, mul ...) without layer initialization
equation_learner = EquationLearner(functions=[Tan, Add, Sin, Mul])
equation_learner_2 = EquationLearner(functions=[Tan, Add, Sin, Mul, Identity])
out = Output('out',equation_learner(x.last()))
out2 = Output('out2',equation_learner_2(inputs=(x.last(),F.last())))
example = Modely(visualizer=None)
example.addModel('model',[out,out2])
example.neuralizeModel()
print(example({'x':[1], 'F':[2]}))

print("------------------------EXAMPLE 6------------------------")
NeuObj.clearNames(['out','out2'])
## Create an equation learner block with functions that take 2 parameters (add, sub, mul ...) with layer initialization
linear_layer_in_1 = Linear(output_dimension=7, W_init=init_constant, W_init_params={'value':1}, b_init=init_constant, b_init_params={'value':0}, b=False) # output dim = 1+2+1+2+1
linear_layer_in_2 = Linear(output_dimension=7, W_init=init_constant, W_init_params={'value':1}, b_init=init_constant, b_init_params={'value':0}, b=False)
equation_learner_1 = EquationLearner(functions=[Tan, Add, Sin, Mul, Identity], linear_in=linear_layer_in_1)
equation_learner_2 = EquationLearner(functions=[Tan, Add, Sin, Mul, Identity], linear_in=linear_layer_in_2)
out = Output('out',equation_learner_1(x.last()))
out2 = Output('out2',equation_learner_2(inputs=(x.last(),F.last())))
example = Modely(visualizer=None)
example.addModel('model',[out,out2])
example.neuralizeModel()
print(example({'x':[1], 'F':[2]}))

print("------------------------EXAMPLE 7------------------------")
NeuObj.clearNames(['out','out2'])
## Create an equation learner block with simple parametric functions
def func1(K1):
    return torch.sin(K1)

def func2(K2):
    return torch.cos(K2)

parfun1 = ParamFun(func1)
parfun2 = ParamFun(func2)
equation_learner = EquationLearner([parfun1, parfun2])

out = Output('out',equation_learner(x.last()))
example = Modely(visualizer=None)
example.addModel('eqlearner',out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))

print("------------------------EXAMPLE 8------------------------")
NeuObj.clearNames(['out','out2'])
## Create an equation learner block with parametric functions that takes parameters
def myFun(K1,p1):
    return K1*p1
K = Parameter('k', dimensions =  1, sw = 1,values=[[2.0]])
parfun = ParamFun(myFun, parameters_and_constants = [K] )

equation_learner = EquationLearner([parfun])

out = Output('out',equation_learner(x.last()))
example = Modely(visualizer=None)
example.addModel('eqlearner',out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
print(example({'x':[1,2],'F':[1,2]}))

print("------------------------EXAMPLE 9------------------------")
NeuObj.clearNames(['out','out2'])
## Create an equation learner block with parametric functions that takes parameters and other functions
def myFun(K1,K2,p1,p2):
    return K1*p1+K2*p2
K1 = Parameter('k1', dimensions =  1, sw = 1,values=[[2.0]])
K2 = Parameter('k2', dimensions =  1, sw = 1,values=[[3.0]])
parfun = ParamFun(myFun, parameters_and_constants = [K1,K2] )

equation_learner = EquationLearner([parfun, Sin, Add])

out = Output('out',equation_learner((x.last(),F.last())))
example = Modely(visualizer=None)
example.addModel('eqlearner',out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
print(example({'x':[1,2],'F':[1,2]}))

print("------------------------EXAMPLE 10------------------------")
NeuObj.clearNames(['out','k'])
## Create an equation learner block with parametric functions and fuzzy
def myFun(K1,p1):
    return K1*p1
K = Parameter('k', dimensions =  1, sw = 1,values=[[2.0]])
parfun = ParamFun(myFun, parameters_and_constants = [K])
fuzzi = Fuzzify(centers=[0,1,2,3])

equation_learner = EquationLearner([parfun, fuzzi])

out = Output('out',equation_learner((x.last(),F.last())))
example = Modely(visualizer=None)
example.addModel('eqlearner',out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
print(example({'x':[1,2],'F':[1,2]}))

print("------------------------EXAMPLE 11------------------------")
NeuObj.clearNames(['out','k1','k2'])
## Create an equation learner block with parametric functions that takes parameters and other functions with temporal windows
def myFun(K1,K2,p1,p2):
    return K1*p1+K2*p2
K1 = Parameter('k1', dimensions =  1, sw = 1,values=[[2.0]])
K2 = Parameter('k2', dimensions =  1, sw = 1,values=[[3.0]])
parfun = ParamFun(myFun, parameters_and_constants = [K1,K2] )

equation_learner = EquationLearner([parfun, Sin, Add])

out = Output('out',equation_learner((x.sw(1),F.sw(1))))
example = Modely(visualizer=None)
example.addModel('eqlearner',out)
example.neuralizeModel()
print(example({'x':[1,2],'F':[1,2]}))
    