import os

from datetime import datetime

from nnodely.visualizer import  Visualizer

class Exporter():

    def __init__(self, workspace = None, visualizer = None, save_history = False):
        # Export parameters
        if workspace is not None:
            self.workspace = workspace
            os.makedirs(self.workspace, exist_ok=True)
            if save_history:
                self.folder = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                self.workspace_folder = os.path.join(self.workspace, self.folder)
            else:
                self.workspace_folder = self.workspace
            os.makedirs(self.workspace_folder, exist_ok=True)

        if visualizer is not None:
            self.visualizer = visualizer
        else:
            self.visualizer = Visualizer()

    def saveTorchModel(self, model, name = 'net', model_folder = None):
        pass

    def loadTorchModel(self, name = 'net', model_folder = None):
        pass

    def saveModel(self, model, name = 'net', model_folder = None):
        pass

    def loadModel(self, name = 'net', model_folder = None):
        pass

    def exportPythonModel(self, name = 'net', model_folder = None):
        pass

    def importPythonModel(self, name = 'net', model_folder = None):
        pass

    def onnx_inference(self, inputs:dict, path:str):
        pass

    def exportReport(self, name = 'net', model_folder = None):
        pass