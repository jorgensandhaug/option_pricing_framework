from abc import ABC, abstractmethod

class SimulationBasedOptionPricingModel(ABC):
    def __init__(self, simulator):
        self.simulator = simulator

    @abstractmethod
    def price(self, params: dict, simulation_params: dict):
        pass
