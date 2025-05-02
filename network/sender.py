import socket
import numpy as np

def send_tensor(tensor_mx, server_ip, port=5001):
    tensor_np = np.array(tensor_mx)  # Convert MLX tensor → NumPy
    tensor_bytes = tensor_np.tobytes()
    shape = tensor_np.shape

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, port))

        # Send shape as line (e.g., "10,10\n")
        shape_line = f"{','.join(map(str, shape))}\n"
        client_socket.sendall(shape_line.encode('utf-8'))

        # Send data
        client_socket.sendall(tensor_bytes)

        print(f"[Sender] Sent tensor of shape {shape} to {server_ip}:{port}")
