import numpy as np
from .distribution_model import DistributionModel

class NormalDistribution(DistributionModel):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def sample(self, size):
        return np.random.normal(self.mean, self.std, size)
