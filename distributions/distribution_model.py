from abc import ABC, abstractmethod

class DistributionModel(ABC):
    @abstractmethod
    def sample(self, size):
        pass