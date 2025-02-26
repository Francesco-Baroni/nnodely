from nnodely.relation import Stream, NeuObj, ToStream
from nnodely.utils import merge, enforce_types

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
    This operation Derivate a Stream

    Parameters
    ----------
    method : is the derivative method
    """
    @enforce_types
    def __init__(self, obj:Stream, method:str = 'ForwardEuler') -> Stream:
        from nnodely.input import State, ClosedLoop
        from nnodely.parameter import SampleTime
        s = State(obj.name + "_der" + str(NeuObj.count), dimensions=obj.dim['dim'])
        if method == 'ForwardEuler':
            new_s = (obj - s.last()) / SampleTime()
        else:
            raise ValueError(f"The method '{method}' is not supported yet")
        out_connect = ClosedLoop(obj, s)
        super().__init__(new_s.name,merge(new_s.json, out_connect.json), new_s.dim)