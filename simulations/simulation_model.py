from abc import ABC, abstractmethod
from distributions.distribution_model import DistributionModel

class SimulationModel(ABC):
    def __init__(self, distribution_model: DistributionModel):
        self.distribution_model = distribution_model

    @abstractmethod
    def simulate(self, params: dict):
        pass
