# Extern packages
import random, torch, copy
import numpy as np

# Main operators
from nnodely.operators.composer import Composer
from nnodely.operators.trainer import Trainer
from nnodely.operators.loader import Loader
from nnodely.operators.validator import Validator
from nnodely.operators.exporter import Exporter

# nnodely packages
from nnodely.visualizer import EmptyVisualizer, TextVisualizer
from nnodely.exporter import EmptyExporter
from nnodely.basic.relation import NeuObj
from nnodely.support.utils import ReadOnlyDict, ParamDict, enforce_types

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)


@enforce_types
def clearNames(names:str|list|None = None):
    NeuObj.clearNames(names)

class Modely(Composer, Trainer, Loader, Validator, Exporter):
    """
    Create the main object, the nnodely object, that will be used to create the network, train and export it.

    Parameters
    ----------
    visualizer : str, Visualizer, optional
        The visualizer to be used. Default is the 'Standard' visualizer.
    exporter : str, Exporter, optional
        The exporter to be used. Default is the 'Standard' exporter.
    seed : int, optional
        Set the seed for all the random modules inside the nnodely framework. Default is None.
    workspace : str
        The path of the workspace where all the exported files will be saved.
    log_internal : bool
        Whether or not save the logs. Default is False.
    save_history : bool
        Whether or not save the history. Default is False.

    Example
    -------
        >>> model = Modely()
    """
    @enforce_types
    def __init__(self,
                 visualizer:str|EmptyVisualizer|None = 'Standard',
                 exporter:str|EmptyExporter|None = 'Standard',
                 seed:int|None = None,
                 workspace:str|None = None,
                 log_internal:bool = False,
                 save_history:bool = False):

        ## Set the random seed for reproducibility
        if seed is not None:
            self.resetSeed(seed)

        # Visualizer
        if visualizer == 'Standard':
            self.visualizer = TextVisualizer(1)
        elif visualizer != None:
            self.visualizer = visualizer
        else:
            self.visualizer = EmptyVisualizer()
        self.visualizer.setModely(self)

        Composer.__init__(self)
        Loader.__init__(self)
        Trainer.__init__(self, log_internal)
        Validator.__init__(self)
        Exporter.__init__(self, exporter, workspace, save_history=save_history)

    @property
    def neuralized(self):
        return self._neuralized

    @neuralized.setter
    def neuralized(self, value):
        raise AttributeError("Cannot modify read-only property 'neuralized' use neuralizeModel() instead.")

    @property
    def traced(self):
        return self._traced

    @traced.setter
    def traced(self, value):
        raise AttributeError("Cannot modify read-only property 'traced'.")

    @property
    def parameters(self):
        if self._neuralized:
            return ParamDict(self._model_def['Parameters'], self._model.all_parameters)
        else:
            return ParamDict(self._model_def['Parameters'])

    @property
    def constants(self):
        return ReadOnlyDict({key:value.detach().numpy().tolist() for key,value in self._model.all_constants})

    @property
    def states(self):
        return {key:value.detach().numpy().tolist() for key,value in self._states.items()}

    @property
    def json(self):
        return copy.deepcopy(self._model_def._ModelDef__json)

    @enforce_types
    def resetSeed(self, seed:int) -> None:
        """
        Resets the random seed for reproducibility.

        This method sets the seed for various random number generators used in the project to ensure reproducibility of results.

        :param seed: The seed value to be used for the random number generators.
        :type seed: int

        Example:
            >>> model = nnodely()
            >>> model.resetSeed(42)
        """
        torch.manual_seed(seed)  ## set the pytorch seed
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)  ## set the random module seed
        np.random.seed(seed)  ## set the numpy seed


nnodely = Modely