import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from nnodely import *

# This example shows how to fit a simple linear model.
# The model chosen is a mass spring damper.
# The data was created previously and loaded from file.
# The data are the position/velocity of the mass and the force applied.
# The neural model mirrors the structure of the physical model.
# The network build estimate the future position of the mass and the velocity.

# Create neural model
# List the input of the model
x = Input('x') # Position of the mass
F = Input('F') # Force

# List the output of the model
out = Fir(parameter_init=init_negexp)(x.tw(0.2))+Fir(parameter_init=init_constant,parameter_init_params={'value':1})(F.last())
xk1 = Output('x[k+1]', out)

# Add the neural models to the nnodely structure
mass_spring_damper = Modely(seed=0)
mass_spring_damper.addModel('xk1',xk1)

# These functions are used to impose the minimization objectives.
# Here it is minimized the error between the future position of x get from the dataset x.z(-1)
# and the estimator designed useing the neural network. The miniminzation is imposed via MSE error.
mass_spring_damper.addMinimize('next-pos', x.next(), xk1, 'mse')


# Nauralize the model and gatting the neural network. The sampling time depends on the datasets.
mass_spring_damper.neuralizeModel(sample_time = 0.05) # The sampling time depends on the dataset

# Data load
data_struct = ['time','x','dx','F']
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','data')
mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

#Neural network train not reccurent training
params = {'num_of_epochs': 100,
          'train_batch_size': 128,
          'lr':0.001}
mass_spring_damper.trainModel(splits=[70,20,10], training_params = params)

vis = MPLNotebookVisualizer()
vis.set_n4m(mass_spring_damper)
#vis.showResult("train_mass_spring_dataset_0.70")

target = Input('target')
measure = Input('measure')
kp = Parameter('P', dimensions=1, values=[[1]])
ki = Parameter('I', dimensions=1, values=[[1]])
kd = Parameter('D', dimensions=1, values=[[1]])
error = target.last()-measure.last()
control = Integrate(error)*ki+error*kp+Derivate(error)*kd
controlForce = Output('PIDForce', control)

mass_spring_damper.removeMinimize('next-pos')
mass_spring_damper.addModel('PID',controlForce)
mass_spring_damper.addMinimize('control-sys', x.next(), target.next(), 'mse')
mass_spring_damper.neuralizeModel()

import numpy as np
data_target = np.ones(100, dtype=np.float32)

dataset = {'target': data_target }
mass_spring_damper.loadData('dataset', dataset)
params = {'num_of_epochs': 5000,
          'train_batch_size': 16,
          'lr':0.2}
mass_spring_damper.trainModel(models='PID',train_dataset= 'dataset', training_params = params, closed_loop={'x':'x[k+1]','measure':'x[k+1]'},connect={'F':'PIDForce'}, prediction_samples=25)

# Add visualizer and show the results on the loaded dataset
vis.showResult("dataset")
