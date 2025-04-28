import json
import re
from collections import defaultdict

def generate_device_map(index_file_path, num_devices):
    with open(index_file_path, 'r') as f:
        index_data = json.load(f)

    param_to_file = index_data.get('weight_map', index_data)  # Handle different formats

    layer_pattern = re.compile(r'model\.layers\.(\d+)\.')

    layer_to_params = defaultdict(list)
    other_params = set()

    for param_name in param_to_file.keys():
        match = layer_pattern.match(param_name)
        if match:
            layer_num = int(match.group(1))
            layer_to_params[layer_num].append(param_name)
        else:
            # Strip '.weight' suffix if present
            base_param = param_name
            if param_name.endswith('.weight'):
                base_param = param_name[:-7]
            other_params.add(base_param)

    sorted_layers = sorted(layer_to_params.keys())
    total_layers = len(sorted_layers)

    layers_per_device = total_layers // num_devices
    remainder = total_layers % num_devices

    device_map = {}

    current_layer = 0
    for device in range(num_devices):
        num_layers = layers_per_device + (1 if device < remainder else 0)
        for _ in range(num_layers):
            if current_layer >= total_layers:
                break
            layer_num = sorted_layers[current_layer]
            layer_key = f"model.layers.{layer_num}"
            device_map[layer_key] = device
            current_layer += 1

    # Assign remaining parameters (e.g., model.norm, lm_head, model.embed_tokens) to the last device
    for param in other_params:
        device_map[param] = num_devices - 1

    return device_map
