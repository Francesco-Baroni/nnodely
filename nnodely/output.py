from pprint import pformat

from nnodely.relation import Stream
from nnodely.utils import check

from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

class Output(Stream):
    """
    Represents an output in the neural network model. This relation is what the network will give as output during inference.

    Parameters
    ----------
    name : str
        The name of the output.
    relation : Stream
        The relation to be used for the output.

    Attributes
    ----------
    name : str
        The name of the output.
    json : dict
        A dictionary containing the configuration of the output.
    dim : dict
        A dictionary containing the dimensions of the output.
    """
    def __init__(self, name, relation):
        """
        Initializes the Output object.

        Parameters
        ----------
        name : str
            The name of the output.
        relation : Stream
            The relation to be used for the output.
        """
        super().__init__(name, relation.json, relation.dim)
        log.debug(f"Output {name}")
        self.json['Outputs'][name] = {}
        self.json['Outputs'][name] = relation.name
        log.debug("\n"+pformat(self.json))

    def closedLoop(self, obj):
        check(False, TypeError,
              f"The {self} must be a Stream and not a {type(self)}.")

    def connect(self, obj):
        check(False, TypeError,
              f"The {self} must be a Stream and not a {type(self)}.")