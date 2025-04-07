# Extern packages
import random, torch
import numpy as np

# Main operators
from nnodely.operators.network import Network
from nnodely.operators.trainer import Trainer
from nnodely.operators.loader import Loader
from nnodely.operators.validator import Validator
from nnodely.operators.exporter import Exporter

# nnodely packages
from nnodely.visualizer import TextVisualizer, Visualizer
from nnodely.basic.modeldef import ModelDef
from nnodely.basic.relation import NeuObj

from nnodely.support.utils import TORCH_DTYPE

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)


def clearNames(names:str|None = None):
    NeuObj.clearNames(names)

class Modely(Network, Trainer, Loader, Validator, Exporter):
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
    def __init__(self,
                 visualizer:str|Visualizer|None = 'Standard',
                 exporter:str|Exporter|None = 'Standard',
                 seed:int|None = None,
                 workspace:str|None = None,
                 log_internal:bool = False,
                 save_history:bool = False):

        # Visualizer
        if visualizer == 'Standard':
            self.visualizer = TextVisualizer(1)
        elif visualizer != None:
            self.visualizer = visualizer
        else:
            self.visualizer = Visualizer()
        self.visualizer.setModely(self)

        ## Set the random seed for reproducibility
        if seed is not None:
            self.resetSeed(seed)

        # Save internal
        self.log_internal = log_internal
        if self.log_internal == True:
            self.internals = {}

        # Models definition
        self.model_def = ModelDef()
        self.model = None
        self.neuralized = False
        self.traced = False

        Network.__init__(self)
        Loader.__init__(self)
        Trainer.__init__(self)
        Validator.__init__(self)
        Exporter.__init__(self, exporter, workspace, save_history)

    def resetSeed(self, seed):
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