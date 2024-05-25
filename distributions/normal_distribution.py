import numpy as np
from .distribution_model import DistributionModel
from scipy.stats import norm

class NormalDistribution(DistributionModel):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std


    def sample(self, size):
        return np.random.normal(self.mean, self.std, size)

    def ppf(self, q):
        return norm.ppf(q, self.mean, self.std)
