from nnodely.relation import NeuObj, AutoToStream, Stream
from nnodely.utils import merge, enforce_types

# Binary operators
int_relation_name = 'Integrate'
der_relation_name = 'Derivate'

class Integrate(NeuObj, AutoToStream):
    """
    This operation Integrate a Stream

    Parameters
    ----------
    method : is the integration method
    """
    @enforce_types
    def __init__(self, method:str = 'ForwardEuler'):
        self.method = method
        super().__init__(int_relation_name + str(NeuObj.count))

    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        from nnodely.input import State, ClosedLoop
        from nnodely.parameter import SampleTime
        s = State(self.name + "_last", dimensions=obj.dim['dim'])
        if self.method == 'ForwardEuler':
            DT = SampleTime()
            new_s = s.last()  + obj * DT
        out_connect = ClosedLoop(new_s, s)
        return Stream(new_s.name, merge(new_s.json, out_connect.json), new_s.dim, 1)

class Derivate(NeuObj, AutoToStream):
    """
    This operation Derivate a Stream

    Parameters
    ----------
    method : is the derivative method
    """
    @enforce_types
    def __init__(self, method:str = 'ForwardEuler'):
        self.method = method
        super().__init__(der_relation_name + str(NeuObj.count))

    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        from nnodely.input import State, ClosedLoop
        from nnodely.parameter import SampleTime
        s = State(self.name + "_last", dimensions=obj.dim['dim'])
        if self.method == 'ForwardEuler':
            DT = SampleTime()
            new_s = (obj - s.last()) / DT
        out_connect = ClosedLoop(obj, s)
        return Stream(new_s.name, merge(new_s.json, out_connect.json), new_s.dim, 1)

