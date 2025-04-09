
__version__ = '1.4.0'

import sys
major, minor = sys.version_info.major, sys.version_info.minor

import logging
LOG_LEVEL = logging.INFO

if major < 3:
    sys.exit("Sorry, Python 2 is not supported. You need Python >= 3.10 for "+__package__+".")
elif minor < 9:
    sys.exit("Sorry, You need Python >= 3.10 for "+__package__+".")
else:
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'+
          f' {__package__}_v{__version__} '.center(20, '-')+
          f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

# Network input, outputs and parameters
from nnodely.layers.input import Input, State, Connect, ClosedLoop
from nnodely.layers.parameter import Parameter, Constant, SampleTime
from nnodely.layers.output import Output

# Network elements
from nnodely.layers.activation import Relu, ELU, Softmax, Sigmoid, Identity
from nnodely.layers.fir import Fir
from nnodely.layers.linear import Linear
from nnodely.layers.arithmetic import Add, Sum, Sub, Mul, Div, Pow, Neg
from nnodely.layers.trigonometric import Sin, Cos, Tan, Cosh, Tanh, Sech
from nnodely.layers.parametricfunction import ParamFun
from nnodely.layers.fuzzify import Fuzzify
from nnodely.layers.part import  Part, Select, Concatenate, SamplePart, SampleSelect, TimePart, TimeConcatenate
from nnodely.layers.localmodel import LocalModel
from nnodely.layers.equationlearner import EquationLearner
from nnodely.layers.timeoperation import Integrate, Derivate
from nnodely.layers.interpolation import Interpolation

# Main nnodely classes
from nnodely.nnodely import nnodely, Modely, clearNames
from nnodely.visualizer import Visualizer, TextVisualizer, MPLVisualizer, MPLNotebookVisualizer
from nnodely.exporter import StandardExporter

# Basic nnodely
from nnodely.basic.optimizer import Optimizer, SGD, Adam

# Support functions
from nnodely.support.initializer import init_negexp, init_lin, init_constant, init_exp