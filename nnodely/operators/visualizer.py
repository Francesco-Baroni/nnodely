from nnodely.visualizer import EmptyVisualizer, TextVisualizer
from nnodely.operators.network import Network
from nnodely.support.utils import  check, TORCH_DTYPE, enforce_types

class Visualizer(Network):
    @enforce_types
    def __init__(self, visualizer:EmptyVisualizer|str|None=None):
        check(type(self) is not Visualizer, TypeError, "Visualizer class cannot be instantiated directly")
        super().__init__()

        # Visualizer
        if visualizer == 'Standard':
            self.visualizer = TextVisualizer(1)
        elif visualizer != None:
            self.visualizer = visualizer
        else:
            self.visualizer = EmptyVisualizer()
        self.visualizer.setModely(self)
