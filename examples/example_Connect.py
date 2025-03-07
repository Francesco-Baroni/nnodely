import sys, os
import numpy as np

from nnodely.relation import NeuObj

sys.path.append(os.getcwd())

from nnodely import *

def linear_function(x, k1, k2):
    return x*k1 + k2

data_a = np.arange(1,101, dtype=np.float32)
data_b_t = linear_function(data_a, 2, 3)

data_c = np.arange(1,101, dtype=np.float32)
data_b_in = np.arange(5,105, dtype=np.float32)
data_d_t = linear_function(data_c, 5, 1)

dataset = {'a': data_a, 'b_t': data_b_t, 'c':data_c, 'b_in': data_b_in, 'd_t':data_d_t }


print("------------------------EXAMPLE 1------------------------")
# Modello b
a = Input('a')
b_t = Input('b_t')
condiviso = Parameter('condiviso',dimensions=(1,1))
b = Output('b',Linear(W=condiviso)(a.last())+Linear(W='A')(Fir(W='B')(a.tw(0.5))))

model = Modely(seed=42,visualizer=MPLVisualizer())
model.addModel('b_model', b)
model.addMinimize('b_min', b, b_t.last())
model.neuralizeModel(0.1)

# Modello d
c = Input('c')
b_in = Input('b_in')
d_t = Input('d_t')
d = Output('d',Linear(W=condiviso)(c.last())+Fir(W='C')(c.tw(0.5))+Fir(W='D')(b_in.tw(0.3)))

model.addModel('d_model', d)
model.addMinimize('d_min', d, d_t.last())
model.neuralizeModel(0.1)
model.loadData('dataset', dataset)

params = {'num_of_epochs': 1,
        'train_batch_size': 8,
        'val_batch_size': 8,
        'test_batch_size':1,
        'lr':0.1}

## training dei parametri di tutti i modelli
model.trainModel(splits=[100,0,0], training_params=params, prediction_samples=4, connect={'b_in':'b'})
print('connect variables: ', model.model.connect_update)
print('closed loop variables: ', model.model.closed_loop_update)


print("------------------------EXAMPLE 2------------------------")
NeuObj.clearNames()
# Modello b
a = Input('a')
b_t = Input('b_t')
condiviso = Parameter('condiviso',dimensions=(1,1))
b = Output('b',Linear(W=condiviso)(a.last())+Linear(W='A')(Fir(W='B')(a.tw(0.5))))

model = Modely(seed=42)
model.addModel('b_model', b)
model.addMinimize('b_min', b, b_t.last())
model.neuralizeModel(0.1)

# Modello d
c = Input('c')
d_t = Input('d_t')
b_in = State('b_in')
model.addConnect(b, b_in)
d = Output('d',Linear(W=condiviso)(c.last())+Fir(W='C')(c.tw(0.5))+Fir(W='D')(b_in.tw(0.3)))

model.addModel('d_model', [b,d])
model.addMinimize('d_min', d, d_t.last())
model.neuralizeModel(0.1)
model.loadData('dataset', dataset)

params = {'num_of_epochs': 1,
        'train_batch_size': 8,
        'val_batch_size': 8,
        'test_batch_size':1,
        'lr':0.1}

## training dei parametri di tutti i modelli
model.trainModel(splits=[100,0,0], training_params=params, prediction_samples=4)
print('connect variables: ', model.model.connect_update)
print('closed loop variables: ', model.model.closed_loop_update)