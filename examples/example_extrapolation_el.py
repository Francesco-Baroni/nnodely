import sys, os

import numpy as np
import torch

from nnodely import *

def sigma(x):
    return 1/(1+np.exp(-x))

def sigma_torch(x):
    return 1/(1+torch.exp(-x))


def func(t, y):
    return -0.024*(t**2) -0.064*(y**2) +0.064*t -0.112*y*t +0.256*y -0.5*np.cos(1.2*t -0.15*y -1.8) -0.2*np.sin(-1.4*y -0.8) +1.3*sigma(1.4*t +0.5*y)

## Create dataset
gamma_values = np.linspace(0, 3, 21)
dataset = {'t': [], 'y': [], 'target': []}
for gamma in gamma_values:
    t_values = np.random.uniform(0, 10, 200)
    for t in t_values:
        dataset['t'].append(t)
        dataset['y'].append(gamma)
        dataset['target'].append(func(t, gamma))

t = Input('t')
y = Input('y')
target = Input('target')

linear_in = Linear(output_dimension=7, b=True)
linear_in_2 = Linear(output_dimension=7, b=True)
linear_out = Linear(output_dimension=1, b=True)
sigma_fun = ParamFun(param_fun=sigma_torch)

equation_learner = EquationLearner(functions=[Identity, Sin, Cos, sigma_fun, Mul, Sech], linear_in=linear_in, linear_out=linear_out)
#equation_learner2 = EquationLearner(functions=[Identity, Sin, Cos, sigma_fun, Mul, Sech],linear_in=linear_in_2, linear_out=linear_out) 

eq1 = equation_learner(inputs=(t.last(), y.last()))
#eq2 = equation_learner2(eq1)
out = Output('out', eq1)

example = Modely(visualizer=TextVisualizer())
example.addModel('equation_learner',[out])
example.addMinimize('error', out, target.last())
example.neuralizeModel()
example.loadData(name='dataset', source=dataset)

## Print the initial weights
print('BEFORE TRAINING')
for key in example.model.all_parameters.keys():
    print(f'{key}: {example.model.all_parameters[key].data.numpy()}')
optimizer_defaults = {'weight_decay': 0.02,}
example.trainModel(train_dataset='dataset', lr=0.05, num_of_epochs=1000, optimizer_defaults=optimizer_defaults)
print('AFTER TRAINING')
for key in example.model.all_parameters.keys():
    print(f'{key}: {example.model.all_parameters[key].data.numpy()}')

## Write the equation
threshold = 0.01
equation = ''
functions = ['Identity', 'Sin', 'Cos', 'sigma_fun', 'Mul', 'Mul', 'Sech']
for value_t, value_y, value_b, function in zip(example.model.all_parameters['PLinear4W'].data.numpy()[0], example.model.all_parameters['PLinear4W'].data.numpy()[1], example.model.all_parameters['PLinear4b'].data.numpy(), functions):
    equation_temp = ''
    if abs(value_t) > threshold and abs(value_y) > threshold:
        equation_temp = f'({value_t:.2f}*t) + ({value_y:.2f}*y)'
    elif abs(value_t) > threshold:
        equation_temp = f'({value_t:.2f}*t)'
    elif abs(value_y) > threshold:
        equation_temp = f'({value_y:.2f}*y)'

    if abs(value_b) > threshold:
            equation_temp += f' + ({value_b:.2f})'

    if equation_temp != '':
        equation += f"{function}( {equation_temp} ) + "

print('Equation: ', equation[:-2])

sample = example.getSamples('dataset')
inference = example(sample, sampled=True)
print(f't: {sample["t"][0]}, y: {sample["y"][0]}, target: {sample["target"][0]}')
print('inference: ', inference)