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
    def derivate(self, obj):
        return (obj - obj.sw([-2,-1])) / self.dt

class Trapezoidal(FixedStepSolver):
    def __init__(self):
        super().__init__()
    def integrate(self, obj):
        return (obj + obj.sw([-2,-1])) * 0.5 * self.dt
    def derivate(self, obj):
        from nnodely.layers.input import State, ClosedLoop
        s = State(obj.name + "_der" + str(NeuObj.count), dimensions=obj.dim['dim'])
        new_s = ((obj - obj.sw([-2, -1])) * 2.0) / self.dt - s.last()
        out = ClosedLoop(new_s, s)
        return out