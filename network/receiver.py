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
