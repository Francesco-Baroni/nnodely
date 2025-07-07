from nnodely.support.logger import logging, nnLogger
from nnodely.support.jsonutils import plot_structure, plot_graphviz_structure

log = nnLogger(__name__, logging.INFO)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[%dm"
COLOR_BOLD_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

def color(msg, color_val = GREEN, bold = False):
    if bold:
        return COLOR_BOLD_SEQ % (30 + color_val) + msg + RESET_SEQ
    return COLOR_SEQ % (30 + color_val) + msg + RESET_SEQ

class EmptyVisualizer:
    def __init__(self):
        pass

    def setModely(self, modely):
        self.modely = modely

    def showModel(self, model):
        pass

    def showaddMinimize(self,variable_name):
        pass

    def showModelInputWindow(self):
        pass

    def showModelRelationSamples(self):
        pass

    def showBuiltModel(self):
        pass

    def showWeights(self, weights = None):
        pass

    def showFunctions(self, functions = None):
        pass

    def showWeightsInTrain(self, batch = None, epoch = None, weights = None):
        pass

    def showDataset(self, name):
        pass

    def showStartTraining(self):
        pass

    def showTraining(self, epoch, train_losses, val_losses):
        pass

    def showEndTraining(self, epoch, train_losses, val_losses):
        pass

    def showTrainParams(self):
        pass

    def showTrainingTime(self, time):
        pass

    def showResult(self, name_data):
        pass

    def showResults(self):
        pass

    def saveModel(self, name, path):
        pass

    def loadModel(self, name, path):
        pass

    def exportModel(self, name, path):
        pass

    def importModel(self, name, path):
        pass

    def exportReport(self, name, path):
        pass

    def plotStructure(self, json=None, filename='nnodely_graph', library='matplotlib'):
        json = self.modely.json if json is None else json
        if json is None:
            raise ValueError("No JSON model definition provided. Please provide a valid JSON model definition.")
        if library not in ['matplotlib', 'graphviz']:
            raise ValueError("Invalid library specified. Use 'matplotlib' or 'graphviz'.")
        if library == 'matplotlib':
            plot_structure(json, filename)
        elif library == 'graphviz':
            plot_graphviz_structure(json, filename)
