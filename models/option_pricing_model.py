from abc import ABC, abstractmethod

class OptionPricingModel(ABC):
    @abstractmethod
    def price(self, params: dict):
        pass
