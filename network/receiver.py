import socket
import numpy as np
import mlx.core as mx

def wait_for_tensor(port=5001, dtype=np.float32):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('', port))
        server_socket.listen(1)
        print(f"[Receiver] Waiting for connection on port {port}...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"[Receiver] Connected by {addr}")

            # Receive shape line (e.g., "10,10\n")
            shape_line = b""
            while not shape_line.endswith(b"\n"):
                shape_line += conn.recv(1)
            shape = tuple(map(int, shape_line.decode('utf-8').strip().split(',')))
            print(f"[Receiver] Expecting tensor of shape {shape}")

            expected_bytes = np.prod(shape) * np.dtype(dtype).itemsize

            # Receive data
            received = b""
            while len(received) < expected_bytes:
                packet = conn.recv(4096)
                if not packet:
                    break
                received += packet

            tensor_np = np.frombuffer(received, dtype=dtype).reshape(shape)
            tensor_mx = mx.array(tensor_np)
            print(f"[Receiver] Received tensor successfully: shape {tensor_mx.shape}")

            return tensor_mx

