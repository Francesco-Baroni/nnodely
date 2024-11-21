# Extern packages
import random, torch
import numpy as np

# nnodely packages

class nnodely:
    """
    Create the main object, the nnodely object, that will be used to create the network, train and export it.

    :param seed: It is the seed used for the random number generator
    :type seed: int or None

    Example:
        >>> model = nnodely()
    """
    def __init__(self, seed:int|None = None):
        ## Set the random seed for reproducibility
        if seed is not None:
            self.resetSeed(seed)

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
