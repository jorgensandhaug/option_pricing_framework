import numpy as np
from scipy.stats import t
from .distribution_model import DistributionModel

class TDistribution(DistributionModel):
    def __init__(self, df, loc=0, scale=1):
        self.df = df
        self.loc = loc
        self.scale = scale

    def sample(self, size):
        return t.rvs(self.df, loc=self.loc, scale=self.scale, size=size)
