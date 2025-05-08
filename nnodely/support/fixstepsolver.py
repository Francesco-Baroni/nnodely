from nnodely.layers.parameter import SampleTime
from nnodely.basic.relation import NeuObj

class FixedStepSolver():
    def __init__(self):
        self.dt = SampleTime()

class Euler(FixedStepSolver):
    def __init__(self):
        super().__init__()
    def integrate(self, obj):
        return obj * self.dt
    def derivate(self, obj, old_obj):
        return (obj - old_obj) / self.dt

class Trapezoidal(FixedStepSolver):
    def __init__(self):
        super().__init__()
    def integrate(self, obj):
        from nnodely.layers.input import Input, Connect
        s = Input(obj.name + "_der" + str(NeuObj.count), dimensions=obj.dim['dim'])
        obj = Connect(obj, s, local=True)
        return (obj + s.sw([-2,-1])) * 0.5 * self.dt
    def derivate(self, obj, old_obj):
        from nnodely.layers.input import Input, ClosedLoop
        s = Input(obj.name + "_der" + str(NeuObj.count), dimensions=obj.dim['dim'])
        new_s = ((obj - old_obj) * 2.0) / self.dt - s.last()
        new_s = ClosedLoop(new_s, s, local=True)
        return new_s