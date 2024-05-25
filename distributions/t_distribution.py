import numpy as np
from scipy.stats import t
from .distribution_model import Distribution

class ScaledTDistribution(Distribution):
    def __init__(self, df, loc=0, scale=1):
        self.df = df
        self.loc = loc
        self.scale = scale

    def sample(self, size):
        return t.rvs(self.df, loc=self.loc, scale=self.scale, size=size)
    
    def ppf(self, q):
        return t.ppf(q, self.df, loc=self.loc, scale=self.scale)

class TDistribution(ScaledTDistribution):
    def __init__(self, df):
        super().__init__(df, loc=0, scale=1)
        


