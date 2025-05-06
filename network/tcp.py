import socket
import numpy as np
import mlx.core as mx

def wait_for_tensor(port=5001):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('', port))
        server_socket.listen(1)
        print(f"[Receiver] Waiting for connection on port {port}...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"[Receiver] Connected by {addr}")

            # Receive header line
            header_line = b""
            while not header_line.endswith(b"\n"):
                header_line += conn.recv(1)
            parts = header_line.decode('utf-8').strip().split(',')
            shape = tuple(map(int, parts[:-1]))
            dtype = np.dtype(parts[-1])

            print(f"[Receiver] Expecting tensor of shape {shape} and dtype {dtype}")

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

            print(f"[Receiver] Received tensor successfully: shape {tensor_mx.shape}, dtype {tensor_mx.dtype}")
            return tensor_mx

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
