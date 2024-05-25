from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def sample(self, size):
        pass

    def ppf(self, q):
        pass
