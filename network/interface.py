from abc import ABC, abstractmethod

class NetworkInterface(ABC):
    @abstractmethod
    def send_tensor(self, tensor, dest_rank=1, tag=0, **kwargs):
        pass

    @abstractmethod
    def wait_for_tensor(self, source_rank=0, tag=0, **kwargs):
        pass
