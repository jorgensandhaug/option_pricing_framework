from abc import ABC, abstractmethod
from distributions.distribution_model import Distribution

class SimulationModel(ABC):
    def __init__(self, simulation_params: dict=None):
        self.simulation_params = simulation_params

    @abstractmethod
    def simulate(self, simulation_params: dict):
        pass
