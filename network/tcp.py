import socket
import numpy as np
import mlx.core as mx
import json, os

from utils.utils import log_debug

# Path to your JSON file
file_path = 'network/nodes.json'

# Convert to absolute path
file_path = os.path.abspath(file_path)

# Open and read the JSON file
with open(file_path, 'r') as file:
    nodes = json.load(file)

def wait_for_tensor(machine, tag=0, port=5001):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('', port))
        server_socket.listen(1)
        log_debug(f"[Receiver] Waiting for connection on port {port}...")

        conn, addr = server_socket.accept()
        with conn:
            log_debug(f"[Receiver] Connected by {addr}")

            # Receive header line
            header_line = b""
            while not header_line.endswith(b"\n"):
                header_line += conn.recv(1)
            parts = header_line.decode('utf-8').strip().split(',')
            shape = tuple(map(int, parts[:-1]))
            dtype = np.dtype(parts[-1])

            log_debug(f"[Receiver] Expecting tensor of shape {shape} and dtype {dtype}")

            expected_bytes = np.prod(shape) * dtype.itemsize

            # Receive data
            received = b""
            while len(received) < expected_bytes:
                packet = conn.recv(4096)
                if not packet:
                    break
                received += packet

            if len(received) != expected_bytes:
                raise ValueError(f"Expected {expected_bytes} bytes, but received {len(received)} bytes")

            tensor_np = np.frombuffer(received, dtype=dtype).reshape(shape)
            tensor_mx = mx.array(tensor_np)

            log_debug(f"[Receiver] Received tensor successfully: shape {tensor_mx.shape}, dtype {tensor_mx.dtype}")
            return tensor_mx

def send_tensor(tensor_mx, node_id, port=5001):
    tensor_np = np.array(tensor_mx)  # MLX tensor → NumPy
    tensor_bytes = tensor_np.tobytes()
    shape = tensor_np.shape
    dtype_name = tensor_np.dtype.name  # e.g., 'float32', 'int64'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((nodes[node_id], port))

        # Send header: shape + dtype, separated by commas, ending with newline
        header = f"{','.join(map(str, shape))},{dtype_name}\n"
        client_socket.sendall(header.encode('utf-8'))

        # Send data
        client_socket.sendall(tensor_bytes)

        log_debug(f"[Sender] Sent tensor of shape {shape} and dtype {dtype_name} to {nodes[node_id]}:{port}")
