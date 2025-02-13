import sys, os, torch, importlib

from torch.fx import symbolic_trace

from pprint import PrettyPrinter

class JsonPrettyPrinter(PrettyPrinter):
    def _format(self, object, *args):
        if isinstance(object, str):
            width = self._width
            self._width = sys.maxsize
            try:
                super()._format(object.replace('\'','_"_'), *args)
            finally:
                self._width = width
        else:
            super()._format(object, *args)

def save_model(model, model_path):
    # Export the dictionary as a JSON file
    with open(model_path, 'w') as json_file:
        # json.dump(self.model_def, json_file, indent=4)
        json_file.write(JsonPrettyPrinter().pformat(model)
                        .replace('\'', '\"')
                        .replace('_"_', '\'')
                        .replace('None', 'null')
                        .replace('False', 'false')
                        .replace('True', 'true'))
        # json_file.write(JsonPrettyPrinter().pformat(model).replace('None','null'))
        # data = json.dumps(self.model_def)
        # json_file.write(pformat(data).replace('\\\\n', '\\n').replace('\'', '').replace('(','').replace(')',''))
        # json_file.write(pformat(data).replace('\'', '\"'))

def load_model(model_path):
    import json
    with open(model_path, 'r', encoding='UTF-8') as file:
        model_def = json.load(file)
    return model_def

def export_python_model(model_def, model, model_path):
    package_name = __package__.split('.')[0]

    # Get the symbolic tracer
    with torch.no_grad():
        trace = symbolic_trace(model)
        #print('TRACED MODEL \n', trace.code)

    ## Standard way to modify the graph
    # # Replace all _tensor_constant variables with their constant values
    # for node in trace.graph.nodes:
    #     if node.op == 'get_attr' and node.target.startswith('_tensor_constant'):
    #         constant_value = getattr(model, node.target).item()
    #         with trace.graph.inserting_after(node):
    #             new_node = trace.graph.create_node('call_function', torch.tensor, (constant_value,))
    #         node.replace_all_uses_with(new_node)
    #         trace.graph.erase_node(node)
    #
    # # Recompile the graph
    # trace.recompile()
    ## Standard way to modify the graph

    attributes = sorted(set([line for line in trace.code.split() if 'self.' in line]))
    #print('attributes: ', attributes)
    #print('model.relation_forward: ', model.relation_forward.SamplePart14.W)
    saved_functions = []

    with open(model_path, 'w') as file:
        #file.write("import torch.nn as nn\n")
        file.write("import torch\n\n")

        ## write the connect wrap function
        file.write(f"def {package_name}_model_connect(data_in, rel, shift: int):\n")
        file.write("    virtual = torch.cat((data_in[:, shift:, :], data_in[:, :shift, :]), dim=1)\n")
        file.write("    virtual[:, -shift:, :] = rel\n")
        file.write("    return virtual\n\n")

        for name in model_def['Functions'].keys():
            if 'Fuzzify' in name:
                if 'slicing' not in saved_functions:
                    #file.write("@torch.fx.wrap\n")
                    file.write(f"def {package_name}_fuzzify_slicing(res, i, x):\n")
                    file.write("    res[:, :, i:i+1] = x\n\n")
                    saved_functions.append('slicing')

                function_name = model_def['Functions'][name]['names']
                function_code = model_def['Functions'][name]['functions']
                if isinstance(function_code, list):
                    for i, fun_code in enumerate(function_code):
                        if fun_code != 'Rectangular' and fun_code != 'Triangular':
                            if function_name[i] not in saved_functions:
                                fun_code = fun_code.replace(f'def {function_name[i]}',
                                                            f'def {package_name}_fuzzify_{function_name[i]}')
                                #file.write("@torch.fx.wrap\n")
                                file.write(fun_code)
                                file.write("\n")
                                saved_functions.append(function_name[i])
                else:
                    if (function_name != 'Rectangular') and (function_name != 'Triangular') and (
                            function_name not in saved_functions):
                        function_code = function_code.replace(f'def {function_name}',
                                                              f'def {package_name}_fuzzify_{function_name}')
                        #file.write("@torch.fx.wrap\n")
                        file.write(function_code)
                        file.write("\n")
                        saved_functions.append(function_name)


            elif 'ParamFun' in name:
                function_name = model_def['Functions'][name]['name']
                # torch.fx.wrap(self.model_def['Functions'][name]['name'])
                if function_name not in saved_functions:
                    code = model_def['Functions'][name]['code']
                    code = code.replace(f'def {function_name}', f'def {package_name}_parametricfunction_{function_name}')
                    file.write(code)
                    file.write("\n")
                    saved_functions.append(function_name)

        file.write("class TracerModel(torch.nn.Module):\n")
        file.write("    def __init__(self):\n")
        file.write("        super().__init__()\n")
        file.write("        self.all_parameters = {}\n")
        file.write("        self.all_constants = {}\n")
        for attr in attributes:
            if 'all_constant' in attr:
                key = attr.split('.')[-1]
                file.write(
                    f"        self.all_constants[\"{key}\"] = torch.tensor({model.all_constants[key].tolist()}, requires_grad=False)\n")
                #file.write(f"        {attr} = torch.tensor({getattr(trace, attr.replace('self.', ''))})\n")
            elif 'relation_forward' in attr:
                key = attr.split('.')[2]
                if 'Fir' in key or 'Linear' in key:
                    if 'weights' in attr.split('.')[3]:
                        param = model_def['Relations'][key][2]
                        value = model.all_parameters[param] #.squeeze(0) if 'Linear' in key else model.all_parameters[param]
                        file.write(
                            f"        self.all_parameters[\"{param}\"] = torch.nn.Parameter(torch.tensor({value.tolist()}), requires_grad=True)\n")
                    elif 'bias' in attr.split('.')[3]:
                        param = model_def['Relations'][key][3]
                        # value = model.all_parameters[param].data.squeeze(0) if 'Linear' in key else model.all_parameters[param].data
                        # value = model.all_parameters[param].data
                        file.write(
                            f"        self.all_parameters[\"{param}\"] = torch.nn.Parameter(torch.tensor({model.all_parameters[param].tolist()}), requires_grad=True)\n")
                    elif 'dropout' in attr.split('.')[3]:
                        param = model_def['Relations'][key][4]
                        file.write(f"        self.{key} = torch.nn.Dropout(p={param})\n")
                    # param = model_def['Relations'][key][2] if 'weights' in attr.split('.')[3] else model_def['Relations'][key][3]
                    # value = model.all_parameters[param].data.squeeze(0) if 'Linear' in key else model.all_parameters[param].data
                    # file.write(f"        self.all_parameters[\"{param}\"] = torch.nn.Parameter(torch.{value}, requires_grad=True)\n")
                elif 'Part' in key or 'Select' in key: # any(element in key for element in ['Part', 'Select']):
                    value = model.relation_forward[key].W
                    temp_value = str(value).replace(')',', requires_grad=False)')
                    file.write(f"        self.all_constants[\"{key}\"] = torch.{temp_value}\n")
            elif 'all_parameters' in attr:
                key = attr.split('.')[-1]
                file.write(
                    f"        self.all_parameters[\"{key}\"] = torch.nn.Parameter(torch.tensor({model.all_parameters[key].tolist()}), requires_grad=True)\n")
            elif '_tensor_constant' in attr:
                key = attr.split('.')[-1]
                file.write(
                    f"        {attr} = torch.tensor({getattr(model,key).item()})\n")

        file.write("        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)\n")
        file.write("        self.all_constants = torch.nn.ParameterDict(self.all_constants)\n\n")
        file.write("    def update(self, closed_loop={}, connect={}):\n")
        file.write("        pass\n")
        # file.write("        self.closed_loop_update = {}\n")
        # file.write("        self.connect_update = {}\n")
        # file.write("        for key, state in self.state_model.items():\n")
        # file.write("            if 'connect' in state.keys():\n")
        # file.write("                self.connect_update[key] = state['connect']\n")
        # file.write("            elif 'closedLoop' in state.keys():\n")
        # file.write("                self.closed_loop_update[key] = state['closedLoop']\n")
        # file.write("        for connect_in, connect_rel in connect.items():\n")
        # file.write("            self.connect_update[connect_in] = self.outputs[connect_rel]\n")
        # file.write("        for close_in, close_rel in closed_loop.items():\n")
        # file.write("            self.closed_loop_update[close_in] = self.outputs[close_rel]\n")

        for line in trace.code.split("\n")[len(saved_functions) + 2:]:
            if 'self.relation_forward' in line:
                if 'Part' in line or 'Select' in line:
                    attribute = [x for x in line.split() if 'self.relation_forward' in x][0].split('.')[2]
                    old_line = f"self.relation_forward.{attribute}.W"
                    new_line = f"self.all_constants.{attribute}"
                    file.write(f"    {line.replace(old_line, new_line)}\n")
                elif 'dropout' in line:
                    attribute = line.split()[0]
                    layer = attribute.split('_')[2].capitalize()
                    old_line = f"self.relation_forward.{layer}.dropout"
                    new_line = f"self.{layer}"
                    file.write(f"    {line.replace(old_line, new_line)}\n")
                else:
                    attribute = line.split()[-1]
                    relation = attribute.split('.')[2]
                    relation_type = attribute.split('.')[3]
                    param = model_def['Relations'][relation][2] if 'weights' == relation_type else \
                    model_def['Relations'][relation][3]
                    new_attribute = f'self.all_parameters.{param}'
                    file.write(f"    {line.replace(attribute, new_attribute)}\n")
            else:
                file.write(f"    {line}\n")

        if model_def['States'] != {}:
            file.write("class RecurrentModel(torch.nn.Module):\n")
            file.write("    def __init__(self):\n")
            file.write("        super().__init__()\n")
            file.write("        self.Cell = TracerModel()\n")
            list_inputs = "        self.inputs = ["
            for key in model_def['Inputs'].keys():
                list_inputs += f"'{key}', "
            list_inputs += "]\n"
            file.write(list_inputs)
            file.write("        self.states = dict()\n")
            file.write("\n")
            file.write("    def forward(self, kwargs):\n")
            file.write("        n_samples = min([kwargs[key].size(0) for key in self.inputs])\n")
            for key in model_def['States'].keys():
                file.write(f"        self.states['{key}'] = kwargs['{key}']\n")
            result_str = ""
            for key, value in model_def['Outputs'].items():
                result_str += f"'{key}':[], "
            file.write(f"        results = {{{result_str}}}\n")
            file.write("        X = dict()\n")
            file.write("        for idx in range(n_samples):\n")
            file.write(f"            for key in self.inputs:\n")
            file.write(f"                X[key] = kwargs[key][idx]\n")
            file.write(f"            for key, value in self.states.items():\n")
            file.write(f"                X[key] = value\n")
            file.write("            out, _, closed_loop, connect = self.Cell(X)\n")
            file.write("            for key, value in results.items():\n")
            file.write("                results[key].append(out[key])\n")
            file.write("            for key, val in closed_loop.items():\n")
            file.write("                shift = val.size(1)\n")
            file.write("                self.states[key] = nnodely_model_connect(self.states[key], val, shift)\n")
            file.write("            for key, value in connect.items():\n") 
            file.write("                self.states[key] = value\n")
            file.write("        return results\n")

def export_pythononnx_model(model_def, model_path, model_onnx_path, input_order=None, outputs_order=None):
    closed_loop_states, connect_states = [], []
    for key, value in model_def['States'].items():
        if 'closedLoop' in value.keys():
            closed_loop_states.append(key)
        if 'connect' in value.keys():
            connect_states.append(key)
    
    model_inputs = input_order if input_order else list(model_def['Inputs'].keys() | model_def['States'].keys())
    model_outputs = outputs_order if outputs_order else list(model_def['Outputs'].keys())
    inputs = [key for key in model_inputs if key not in model_def['States'].keys()]

    # Define the mapping dictionary input
    trace_mapping_input = {}
    forward = 'def forward(self,'
    for i, key in enumerate(model_inputs):
        value = f'kwargs[\'{key}\']'
        trace_mapping_input[value] = key
        forward = forward + f' {key}' + (',' if i < len(model_inputs) - 1 else '')
    forward = forward + '):'
    # Define the mapping dictionary output
    outputs = '        return ('
    for i, key in enumerate(model_outputs):
        outputs += f'outputs[0][\'{key}\']' + (',' if i < len(model_outputs) - 1 else ',)')
    outputs += ', ('
    for key in closed_loop_states:
        outputs += f'outputs[2][\'{key}\'], '
    outputs += '), ('
    for key in connect_states:
        outputs += f'outputs[3][\'{key}\'], '
    outputs += ')\n'

    # Open and read the file
    file_content = []
    with open(model_path, 'r') as file:
        #file_content = file.read()
        for line in file:
            if 'return ({' in line:
                file_content.append(line)
                break
            file_content.append(line)
        #file_content = file.readlines()[:-26]
    file_content = ''.join(file_content)
    
    # Replace the forward header
    file_content = file_content.replace('def forward(self, kwargs):', forward)
    # Perform the substitution
    for key, value in trace_mapping_input.items():
        file_content = file_content.replace(key, value)
    # Write the modified content back to a new file
    # Replace the return statement
    # Trova l'ultima occorrenza di 'return'
    last_return_index = file_content.rfind('return')
    # Se 'return' è trovato, sostituiscilo con 'outputs ='
    if last_return_index != -1:
        file_content = file_content[:last_return_index] + 'outputs =' + file_content[last_return_index + len('return'):]
    file_content += outputs
    with open(model_onnx_path, 'w') as file:
        file.write(file_content)

        if model_def['States'] != {}:
            file.write('\n')
            file.write("class RecurrentModel(torch.nn.Module):\n")
            file.write("    def __init__(self):\n")
            file.write("        super().__init__()\n")
            file.write("        self.Cell = TracerModel()\n")

            forward_str = "    def forward(self, "
            for key in model_inputs:
                forward_str += f"{key}, "
            forward_str += "):\n"
            file.write(forward_str)

            if model_inputs:
                file.write("        n_samples = min([" + ", ".join([f"{key}.size(0)" for key in inputs]) + "])\n")
            else:
                file.write("        n_samples = 1\n")
            
            for key in model_outputs:
                file.write(f"        results_{key} = []\n")
            file.write("        for idx in range(n_samples):\n")
            call_str = "            out, closed_loop, connect = self.Cell("
            for key in model_inputs:
                call_str += f"{key}[idx], " if key in inputs else f"{key}, "
            call_str += ")\n"
            file.write(call_str)
            #if len(outputs_order) > 1:
            for idx, key in enumerate(model_outputs):
                file.write(f"            results_{key}.append(out[{idx}])\n")
            #else:
            #    file.write(f"            results_{outputs_order[0]}.append(out)\n")
            for idx, key in enumerate(closed_loop_states):
                file.write(f"            shift = closed_loop[{idx}].size(1)\n")
                file.write(f"            {key} = nnodely_model_connect({key}, closed_loop[{idx}], shift)\n")
            for idx, key in enumerate(connect_states):
                file.write(f"            {key} = connect[{idx}]\n")
            for idx, key in enumerate(model_outputs):
                file.write(f"        results_{key} = torch.stack(results_{key}, dim=0)\n")
            #file.write("        results = torch.cat(results, dim=0)\n")
            return_str = "        return "
            for key in model_outputs:
                return_str += f"results_{key}, "
            file.write(return_str)

def import_python_model(name, model_folder):
    sys.path.insert(0, model_folder)
    module_name = os.path.basename(name)
    if module_name in sys.modules:
        # Reload the module if it is already loaded
        module = importlib.reload(sys.modules[module_name])
    else:
        # Import the module if it is not loaded
        module = importlib.import_module(module_name)
    return module.TracerModel()

def export_onnx_model(model_def, model, model_path, input_order=None, output_order=None, name='net_onnx'):
    sys.path.insert(0, model_path)
    module_name = os.path.basename(name)
    if module_name in sys.modules:
        # Reload the module if it is already loaded
        module = importlib.reload(sys.modules[module_name])
    else:
        # Import the module if it is not loaded
        module = importlib.import_module(module_name)
    model = torch.jit.script(module.RecurrentModel()) if model_def['States'] != {} else module.TracerModel()
    model.eval()
    dummy_inputs = []
    input_names = []
    dynamic_axes = {}
    onnx_inputs = input_order if input_order else list(model_def['Inputs'].keys() | model_def['States'].keys())
    for key in onnx_inputs:
        input_names.append(key)
        window_size = model_def['Inputs'][key]['ntot'] if key in model_def['Inputs'].keys() else model_def['States'][key]['ntot']
        dim = model_def['Inputs'][key]['dim'] if key in model_def['Inputs'].keys() else model_def['States'][key]['dim']
        if model_def['States'] != {}:
            if key in model_def['Inputs'].keys():
                dummy_inputs.append(torch.randn(size=(1, 1, window_size, dim)))
                dynamic_axes[key] = {0: 'horizon', 1: 'batch_size'}
            elif key in model_def['States'].keys():
                dummy_inputs.append(torch.randn(size=(1, window_size, dim)))
                dynamic_axes[key] = {0: 'batch_size'}
        else:
            dummy_inputs.append(torch.randn(size=(1, window_size, dim)))
            dynamic_axes[key] = {0: 'batch_size'}
    output_names = output_order if output_order else list(model_def['Outputs'].keys())
    dummy_inputs = tuple(dummy_inputs)

    torch.onnx.export(
        model,                                  # The model to be exported
        dummy_inputs,                           # Tuple of inputs to match the forward signature
        model_path,                             # File path to save the ONNX model
        export_params = True,                   # Store the trained parameters in the model file
        opset_version = 17,                     # ONNX version to export to (you can use 11 or higher)
        do_constant_folding=False,               # Optimize constant folding for inference
        input_names = input_names,              # Name each input as they will appear in ONNX
        output_names = output_names,            # Name the output
        dynamic_axes = dynamic_axes,
    )

def onnx_inference(inputs, path, optimize_graph=False):
    import onnxruntime as ort
    # Create an ONNX Runtime session
    # Define session options
    if optimize_graph == False: ## TODO: Warning when using constant folding in inference CanUpdateImplicitInputNameInSubgraphs]  Implicit input name Cell.all_constants.Constant75 cannot be safely updated to Cell.all_constants.Constant76 in one of the subgraphs.
        session_options = ort.SessionOptions()
        # Set graph optimization level to disable all optimizations
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(path, sess_options=session_options)
    else:
        session = ort.InferenceSession(path)
    output_data = []
    for item in session.get_outputs():
        output_data.append(item.name)
    input_data = {}
    for item in session.get_inputs():
        input_data[item.name] = inputs[item.name]
    # Run inference
    return session.run(output_data, input_data)

def import_onnx_model(name, model_folder):
    import onnxruntime as ort
    model_path = os.path.join(model_folder, name + '.onnx')
    return ort.InferenceSession(model_path)

