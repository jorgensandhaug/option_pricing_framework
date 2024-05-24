import numpy as np
from .simulation_model import SimulationModel

class GeometricBrownianMotion(SimulationModel):
    def simulate(self, params: dict):
        S0 = params['initial_stock_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        sigma = params['volatility']
        delta = params.get('dividend_yield', 0.0)
        steps = params.get('time_steps', 1000)
        simulations = params.get('simulation_paths', 10000)

        dt = T / steps
        prices = np.zeros((simulations, steps + 1))
        prices[:, 0] = S0
        for t in range(1, steps + 1):
            z = self.distribution_model.sample(simulations)
            prices[:, t] = prices[:, t - 1] * np.exp((r - delta - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        return prices
