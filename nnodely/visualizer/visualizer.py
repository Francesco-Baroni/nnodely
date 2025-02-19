from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[%dm"
COLOR_BOLD_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

def color(msg, color_val = GREEN, bold = False):
    if bold:
        return COLOR_BOLD_SEQ % (30 + color_val) + msg + RESET_SEQ
    return COLOR_SEQ % (30 + color_val) + msg + RESET_SEQ

class Visualizer():
    def __init__(self):
        pass

    def set_n4m(self, n4m):
        self.n4m = n4m

    def showModel(self, model):
        pass

    def showStructure(self, json):
        layer_positions = {}
        x, y = 0, 0  # Initial position
        dy, dx = 1.5, 2.5  # Spacing

        ## Layer Inputs: 
        for input_name in json['Inputs'].keys():
            layer_positions[input_name] = (x, y)
            y -= dy
        for state_name in json['States'].keys():
            layer_positions[state_name] = (x, y)
            y -= dy
        for constant_name in json['Constants'].keys():
            layer_positions[constant_name] = (x, y)
            y -= dy
        y_limit = abs(y)

        # Layers Relations:
        available_inputs = list(json['Inputs'].keys() | json['States'].keys() | json['Constants'].keys())
        available_outputs = list(set(json['Outputs'].values()))
        while available_outputs:
            x += dx
            y = 0
            inputs_to_add, outputs_to_remove = [], []
            for relation_name, (relation_type, dependencies, *_) in json['Relations'].items():
                if all(dep in available_inputs for dep in dependencies) and (relation_name not in available_inputs):
                    inputs_to_add.append(relation_name)
                    if relation_name in available_outputs:
                        outputs_to_remove.append(relation_name)
                    layer_positions[relation_name] = (x, y)
                    y -= dy
            y_limit = max(y_limit, abs(y))
            available_inputs.extend(inputs_to_add)
            available_outputs = [out for out in available_outputs if out not in outputs_to_remove]

        ## Layer Outputs: 
        x += dx
        y = 0
        for idx, output_name in enumerate(json['Outputs'].keys()):
            layer_positions[output_name] = (x, y)
            y -= dy  # Move down for the next input
        x_limit = abs(x)
        y_limit = max(y_limit, abs(y))

        # Create the plot
        fig, ax = plt.subplots(figsize=(x_limit, y_limit))
        #fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Plot rectangles for each layer
        colors, labels = ['lightgreen', 'green', 'lightblue', 'orange', 'lightgray'], ['Inputs', 'States', 'Relations', 'Outputs', 'Constants']
        legend_info = [patches.Patch(facecolor=color, edgecolor='black', label=label) for color, label in zip(colors, labels)]
        for layer in (json['Inputs'].keys() | json['States'].keys() | json['Outputs'].keys() | json['Relations'].keys() | json['Constants'].keys()):
            x1, y1 = layer_positions[layer]
            if layer in json['Inputs'].keys():
                color = 'lightgreen'
                tag = f'{layer}\ndim: {json["Inputs"][layer]["dim"]}\nWindow: {json["Inputs"][layer]["ntot"]}'
            elif layer in json['States'].keys():
                color = 'green'
                tag = f'{layer}\ndim: {json["States"][layer]["dim"]}\nWindow: {json["States"][layer]["ntot"]}'
            elif layer in json['Outputs'].keys():
                color = 'orange'
                tag = layer
            elif layer in json['Constants'].keys():
                color = 'lightgray'
                tag = f'{layer}\ndim: {json["Constants"][layer]["dim"]}'
            else:
                color = 'lightblue'
                tag = f'{json["Relations"][layer][0]}\n({layer})'
            rect = patches.Rectangle((x1, y1), 2, 1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(x1 + 1, y1 + 0.5, f"{tag}", ha='center', va='center', fontsize=8, fontweight='bold')

        # Draw arrows for dependencies
        for layer, (_, dependencies, *_) in json['Relations'].items():
            x1, y1 = layer_positions[layer]  # Get position of the current layer
            for dep in dependencies:
                if dep in layer_positions:
                    x2, y2 = layer_positions[dep]  # Get position of the dependent layer
                    ax.annotate("", xy=(x1, y1), xytext=(x2 + 2, y2 + 0.5),
                                arrowprops=dict(arrowstyle="->", color='black', lw=1))
        for out_name, rel_name in json['Outputs'].items():
            x1, y1 = layer_positions[out_name]
            x2, y2 = layer_positions[rel_name]
            ax.annotate("", xy=(x1, y1 + 0.5), xytext=(x2 + 2, y2 + 0.5),
                        arrowprops=dict(arrowstyle="->", color='black', lw=1))
        for key, state in json['States'].items():
            if 'closedLoop' in state.keys():
                x1, y1 = layer_positions[key]
                x2, y2 = layer_positions[state['closedLoop']]
                #ax.annotate("", xy=(x2+1, y2), xytext=(x2+1, y_limit), arrowprops=dict(arrowstyle="-", color='red', lw=1, linestyle='dashed'))
                ax.add_patch(patches.FancyArrowPatch((x2+1, y2), (x2+1, -y_limit), arrowstyle='-', mutation_scale=15, color='red', linestyle='dashed'))
                ax.add_patch(patches.FancyArrowPatch((x2+1, -y_limit), (x1-1, -y_limit), arrowstyle='-', mutation_scale=15, color='red', linestyle='dashed'))
                ax.add_patch(patches.FancyArrowPatch((x1-1, -y_limit), (x1-1, y1+0.5), arrowstyle='-', mutation_scale=15, color='red', linestyle='dashed'))
                ax.add_patch(patches.FancyArrowPatch((x1-1, y1+0.5), (x1, y1+0.5), arrowstyle='->', mutation_scale=15, color='red', linestyle='dashed'))
            elif 'connect' in state.keys():
                x1, y1 = layer_positions[key]
                x2, y2 = layer_positions[state['connect']]
                ax.add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15, color='green', linestyle='dashed'))
            
        legend_info.extend([Line2D([0], [0], color='black', lw=2, label='Dependency'),
                            Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Closed Loop'),
                            Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='Connect')])

        # Adjust the plot limits
        ax.set_xlim(-dx, x_limit+dx)
        ax.set_ylim(-y_limit, dy)
        ax.set_aspect('equal')
        ax.legend(handles=legend_info, loc='lower right')
        ax.axis('off')  # Hide axes

        plt.title(f"Neural Network Diagram - Sampling [{json['Info']['SampleTime']}]", fontsize=12, fontweight='bold')
        plt.show()

    def GraphVizStructure(self, json):
        import graphviz
        return

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