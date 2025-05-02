# import json
# import re
# from collections import defaultdict
#
# def generate_device_map(index_json_path, num_devices):
#     with open(index_json_path, 'r') as f:
#         index_data = json.load(f)
#
#     weight_map = index_data.get("weight_map", {})
#     layer_to_file = defaultdict(set)
#
#     # Extract unique layer prefixes
#     for key, filename in weight_map.items():
#         # Match 'model.layers.N' or other top-level components
#         match = re.match(r'(model\.layers\.\d+)', key)
#         if match:
#             layer_name = match.group(1)
#         else:
#             # For components like 'model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'
#             layer_name = key.split('.')[0] if '.' in key else key
#         layer_to_file[layer_name].add(filename)
#
#     # Sort layers numerically
#     sorted_layers = sorted(layer_to_file.keys(), key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf'))
#
#     device_map = {}
#     for idx, layer in enumerate(sorted_layers):
#         device = idx % num_devices
#         device_map[layer] = device
#
#     return device_map

import json
import re
from collections import defaultdict

def generate_device_map(index_json_path, mps_device='mps', num_mps_layers=35):
    with open(index_json_path, 'r') as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    layer_to_file = defaultdict(set)

    # Extract unique layer prefixes
    for key, filename in weight_map.items():
        # Match 'model.layers.N' or other top-level components
        match = re.match(r'(model\.layers\.\d+)', key)
        if match:
            layer_name = match.group(1)
        else:
            # For components like 'model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'
            layer_name = key.split('.')[0] if '.' in key else key
        layer_to_file[layer_name].add(filename)

    # Sort layers numerically
    sorted_layers = sorted(
        layer_to_file.keys(),
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
    )

    device_map = {}
    for idx, layer in enumerate(sorted_layers):
        if idx < num_mps_layers:
            device_map[layer] = mps_device
        else:
            device_map[layer] = "disk"  # Layers beyond num_mps_layers are not assigned to any device

    return device_map