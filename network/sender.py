import socket
import numpy as np
import mlx.core as mx


def send_tensor(tensor_mx, server_ip, port=5001):
    tensor_np = np.array(tensor_mx)  # MLX tensor → NumPy
    tensor_bytes = tensor_np.tobytes()
    shape = tensor_np.shape
    dtype_name = tensor_np.dtype.name  # e.g., 'float32', 'int64'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, port))

        # Send header: shape + dtype, separated by commas, ending with newline
        header = f"{','.join(map(str, shape))},{dtype_name}\n"
        client_socket.sendall(header.encode('utf-8'))

        # Send data
        client_socket.sendall(tensor_bytes)

        print(f"[Sender] Sent tensor of shape {shape} and dtype {dtype_name} to {server_ip}:{port}")
