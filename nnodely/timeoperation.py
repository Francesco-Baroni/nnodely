import torch.nn as nn
import torch

from nnodely.relation import Stream, NeuObj, ToStream
from nnodely.utils import merge, enforce_types
from nnodely.model import Model

# Binary operators
int_relation_name = 'Integrate'
der_relation_name = 'Derivate'

class Integrate(Stream, ToStream):
    """
    This operation Integrate a Stream

    Parameters
    ----------
    method : is the integration method
    """
    @enforce_types
    def __init__(self, obj:Stream, method:str = 'ForwardEuler') -> Stream:
        from nnodely.input import State, ClosedLoop
        from nnodely.parameter import SampleTime
        s = State(obj.name + "_int" + str(NeuObj.count), dimensions=obj.dim['dim'])
        if method == 'ForwardEuler':
            new_s = s.last()  + obj * SampleTime()
        else:
            raise ValueError(f"The method '{method}' is not supported yet")
        out_connect = ClosedLoop(new_s, s)
        super().__init__(new_s.name, merge(new_s.json, out_connect.json), new_s.dim)

class Derivate(Stream, ToStream):
    """
    This operation Derivate a Stream with respect to time or another Stream

    Parameters
    ----------
    method : is the derivative method
    """
    @enforce_types
    def __init__(self, output:Stream, input:Stream = None, method:str = 'ForwardEuler') -> Stream:
        from nnodely.input import State, ClosedLoop
        from nnodely.parameter import SampleTime
        if input is None:
            s = State(output.name + "_der" + str(NeuObj.count), dimensions=output.dim['dim'])
            if method == 'ForwardEuler':
                new_s = (output - s.last()) / SampleTime()
            else:
                raise ValueError(f"The method '{method}' is not supported yet")
            out_connect = ClosedLoop(output, s)
            super().__init__(new_s.name,merge(new_s.json, out_connect.json), new_s.dim)
        else:
            super().__init__(der_relation_name + str(Stream.count), merge(output.json,input.json), input.dim)
            self.json['Relations'][self.name] = [der_relation_name, [output.name, input.name]]


class Derivate_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Derivate_Layer, self).__init__()

    def forward(self, *inputs):
        return torch.autograd.grad(inputs[0], inputs[1], grad_outputs=torch.ones_like(inputs[0]), create_graph=True, retain_graph=True, allow_unused=False)[0]

def createAdd(name, *inputs):
    #: :noindex:
    return Derivate_Layer()

setattr(Model, der_relation_name, createAdd)
