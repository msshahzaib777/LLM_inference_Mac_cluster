import torch
import torch.distributed.rpc as rpc
from model_split import DeepSeekPart2

class Worker2:
    def __init__(self):
        self.model = DeepSeekPart2()

    def forward(self, hidden_states, input_len):
        return self.model.forward(hidden_states, input_len)

if __name__ == "__main__":
    rpc.init_rpc("worker2", rank=1, world_size=3)
    print("Worker2 ready.")
    import time; time.sleep(999999)
