import pandas as pd
from arch import arch_model

class GARCHModel:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None

    def fit(self, data: pd.Series):
        self.model = arch_model(data, vol='Garch', p=self.p, q=self.q, rescale=False)
        self.fitted_model = self.model.fit(disp='off')
        return self.fitted_model

    def forecast(self, horizon=5):
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting.")
        return np.sqrt(self.fitted_model.forecast(horizon=horizon).variance.values)

