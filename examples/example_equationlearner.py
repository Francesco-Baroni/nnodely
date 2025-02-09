import sys, os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import torch
from nnodely import *

example = 6

x = Input('x')
F = Input('F')

if example == 1:
    print("------------------------EXAMPLE 1------------------------")
    ## Create an equation learner block with pure nnodely functions
    linear_layer = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)(x.last())
    cos_layer = Cos(Select(linear_layer,0))
    sin_layer = Sin(Select(linear_layer,1))
    tan_layer = Tan(Select(linear_layer,2))
    concatenate_layer = Concatenate(Concatenate(tan_layer,sin_layer),cos_layer)

    out_linear = Output('out_linear',linear_layer)
    out_cos = Output('out_cos',cos_layer)
    out_sin = Output('out_sin',sin_layer)
    out_tan = Output('out_tan',tan_layer)
    out_concatenate = Output('out_concatenate',concatenate_layer)
    example = Modely(visualizer=TextVisualizer())
    example.addModel('model',[out_linear,out_cos,out_sin,out_tan,out_concatenate])
    example.neuralizeModel()
    print(example({'x':[1]}))
    
elif example == 2:
    print("------------------------EXAMPLE 2------------------------")
    ## Create an equation learner block similar to Example 1 but using EquationLearner function
    equation_learner = EquationLearner(functions=[Tan, Sin, Cos])
    out = Output('out',equation_learner(x.last()))
    example = Modely(visualizer=TextVisualizer())
    example.addModel('model',[out])
    example.neuralizeModel()
    print(example({'x':[1]}))

elif example == 3:
    print("------------------------EXAMPLE 3------------------------")
    ## Create an equation learner block similar to Example 1 but using EquationLearner function and passing a the linear layer as input
    linear_layer = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)
    equation_learner = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer)
    out = Output('out',equation_learner(x.last()))
    example = Modely(visualizer=MPLVisualizer())
    example.addModel('model',[out])
    example.neuralizeModel()
    print(example({'x':[1]}))

elif example == 4:
    print("------------------------EXAMPLE 4------------------------")
    ## Create an equation learner block similar to Example 1 but using EquationLearner function and passing a the linear layer as input and a linear layer as output
    linear_layer_in = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)
    linear_layer_out = Linear(output_dimension=1, W_init=init_constant, W_init_params={'value':1}, b_init=init_constant, b_init_params={'value':0}, b=False)
    equation_learner = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer_in, linear_out=linear_layer_out)
    out = Output('out',equation_learner(x.last()))
    example = Modely(visualizer=MPLVisualizer())
    example.addModel('model',[out])
    example.neuralizeModel()
    print(example({'x':[1]}))

elif example == 5:
    print("------------------------EXAMPLE 5------------------------")
    ## Create an equation learner block with functions that take 2 parameters (add, sub, mul ...) without layer initialization
    equation_learner = EquationLearner(functions=[Tan, Add, Sin, Mul])
    out = Output('out',equation_learner(x.last()))
    example = Modely(visualizer=MPLVisualizer())
    example.addModel('model',[out])
    example.neuralizeModel()
    print(example({'x':[1]}))

elif example == 6:
    print("------------------------EXAMPLE 6------------------------")
    ## Create an equation learner block with functions that take 2 parameters (add, sub, mul ...) with layer initialization
    linear_layer_in = Linear(output_dimension=6, W_init=init_constant, W_init_params={'value':1}, b_init=init_constant, b_init_params={'value':0}, b=False) # output dim = 1+2+1+2
    equation_learner = EquationLearner(functions=[Tan, Add, Sin, Mul], linear_in=linear_layer_in)
    out = Output('out',equation_learner(x.last()))
    example = Modely(visualizer=MPLVisualizer())
    example.addModel('model',[out])
    example.neuralizeModel()
    print(example({'x':[1]}))

elif example == 7:
    print("------------------------EXAMPLE 7------------------------")
    ## Create an equation learner block with parametric functions
    def func1(K1):
        return torch.sin(K1)

    def func2(K2):
        return torch.cos(K2)

    parfun1 = ParamFun(func1)
    parfun2 = ParamFun(func2)
    equation_learner = EquationLearner([parfun1, parfun2])

    out = Output('out',equation_learner(x.last()))
    example = Modely(visualizer=MPLVisualizer())
    example.addModel('eqlearner',out)
    example.neuralizeModel()
    print(example({'x':[1],'F':[1]}))
    print(example({'x':[1,2],'F':[1,2]}))


elif example == 10:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Given JSON dictionary
    nn_dict = {
        'Add9': ['Add', ['Select7', 'Select8']],
        'Concatenate10': ['Concatenate', ['Tan6', 'Add9']],
        'Concatenate13': ['Concatenate', ['Concatenate10', 'Sin12']],
        'Concatenate17': ['Concatenate', ['Concatenate13', 'Mul16']],
        'Linear4': ['Linear', ['SamplePart3'], 'PLinear3W', None, 0],
        'Mul16': ['Mul', ['Select14', 'Select15']],
        'SamplePart3': ['SamplePart', ['x'], [-1, 0]],
        'Select11': ['Select', ['Linear4'], 0],
        'Select14': ['Select', ['Linear4'], 0],
        'Select15': ['Select', ['Linear4'], 1],
        'Select5': ['Select', ['Linear4'], 0],
        'Select7': ['Select', ['Linear4'], 0],
        'Select8': ['Select', ['Linear4'], 1],
        'Sin12': ['Sin', ['Select11']],
        'Tan6': ['Tan', ['Select5']]
    }

    # Compute positions for layers
    layer_positions = {}
    x, y = 0, 0  # Initial position

    # Assign positions to layers in a simple grid pattern
    for idx, layer in enumerate(nn_dict.keys()):
        layer_positions[layer] = (x, y)
        y -= 1.5  # Move down for the next layer

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot rectangles for each layer
    for layer, (layer_type, dependencies, *_) in nn_dict.items():
        x, y = layer_positions[layer]
        rect = patches.Rectangle((x, y), 2, 1, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        ax.text(x + 1, y + 0.5, f"{layer}\n({layer_type})", ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw arrows for dependencies
    for layer, (_, dependencies, *_) in nn_dict.items():
        x1, y1 = layer_positions[layer]  # Get position of the current layer
        for dep in dependencies:
            if dep in layer_positions:
                x2, y2 = layer_positions[dep]  # Get position of the dependent layer
                ax.annotate("", xy=(x1, y1), xytext=(x2 + 2, y2 + 0.5),
                            arrowprops=dict(arrowstyle="->", color='black', lw=1))

    # Adjust the plot limits
    ax.set_xlim(-1, 6)
    ax.set_ylim(y - 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes

    plt.title("Neural Network Diagram", fontsize=12, fontweight='bold')
    plt.show()