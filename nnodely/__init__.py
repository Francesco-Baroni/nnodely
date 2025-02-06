
__version__ = '0.25.0'

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
from nnodely.input import Input, State, Connect, ClosedLoop
from nnodely.parameter import Parameter, Constant
from nnodely.output import Output

# Network elements
from nnodely.activation import Relu, Tanh, ELU, Sigma, Identity
from nnodely.fir import Fir
from nnodely.linear import Linear
from nnodely.arithmetic import Add, Sum, Sub, Mul, Pow, Neg, Concatenate
from nnodely.trigonometric import Sin, Cos, Tan, Cosh, Sech
from nnodely.parametricfunction import ParamFun
from nnodely.fuzzify import Fuzzify
from nnodely.part import TimePart, TimeSelect, SamplePart, SampleSelect, Part, Select, TimeConcatenate
from nnodely.localmodel import LocalModel
from nnodely.equationlearner import EquationLearner

# Main nnodely classes
from nnodely.nnodely import nnodely, Modely
from nnodely.visualizer import Visualizer, TextVisualizer, MPLVisualizer, MPLNotebookVisualizer
from nnodely.optimizer import Optimizer, SGD, Adam
from nnodely.exporter import Exporter, StandardExporter

# Support functions
from nnodely.initializer import init_negexp, init_lin, init_constant