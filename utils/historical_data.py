import numpy as np
from scipy.stats import t, norm

class HistoricalData:
    def __init__(self, price_data):
        self.price_data = price_data

    def calculate_log_returns(self):
        log_returns = np.diff(np.log(self.price_data))
        return log_returns

    def fit_normal_distribution(self):
        log_returns = self.calculate_log_returns()
        mean, std = norm.fit(log_returns)
        return mean, std

    def fit_t_distribution(self):
        log_returns = self.calculate_log_returns()
        df, loc, scale = t.fit(log_returns)
        return df, loc, scale
