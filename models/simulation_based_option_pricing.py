from abc import ABC, abstractmethod
from typing import Optional



class SimulationBasedOptionPricingModel(ABC):
    def __init__(self, simulator):
        self.simulator = simulator

    @abstractmethod
    def price(self, params: dict, simulation_params: Optional[dict] = None):
        pass
